"""
PT of the City × COB — Quality Procedures Chatbot
──────────────────────────────────────────────────
Run with:  streamlit run chatbot_app.py
"""
 
import os, json, pickle
import streamlit as st
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
 
# ─────────────────────────────────────────────
# CONFIG  — reads from .streamlit/secrets.toml
# ─────────────────────────────────────────────
GROQ_API_KEY  = st.secrets["GROQ_API_KEY"]
APP_PASSWORD  = st.secrets["APP_PASSWORD"]
 
# Try multiple possible locations for data files
BASE_DIR = os.path.dirname(__file__) or "."
POSSIBLE_DATA_DIRS = [
    os.path.join(BASE_DIR, "data"),  # data/ subdirectory
    BASE_DIR,                         # root directory
]

# Find the correct data directory
DATA_DIR = None
for dir_path in POSSIBLE_DATA_DIRS:
    text_path = os.path.join(dir_path, "all_text.json")
    ocr_path = os.path.join(dir_path, "all_ocr.json")
    if os.path.exists(text_path) or os.path.exists(ocr_path):
        DATA_DIR = dir_path
        break

if DATA_DIR is None:
    DATA_DIR = os.path.join(BASE_DIR, "data")  # Default fallback

INDEX_PATH    = os.path.join(DATA_DIR, "faiss_index.pkl")
TEXT_JSON     = os.path.join(DATA_DIR, "all_text.json")
OCR_JSON      = os.path.join(DATA_DIR, "all_ocr.json")
 
GROQ_MODEL    = "llama-3.3-70b-versatile"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K         = 6
# ─────────────────────────────────────────────
 
 
# ══════════════════════════════════════════════
# PASSWORD GATE
# ══════════════════════════════════════════════
def check_password():
    """Returns True if the user has entered the correct password."""
 
    if st.session_state.get("authenticated"):
        return True
 
    # ── login page styling ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
    #MainMenu, footer, header, .stDeployButton,
    [data-testid="stToolbar"] { display: none !important; }
    .stApp { background: linear-gradient(135deg, #0D2137 0%, #0F3460 60%, #0E4D7B 100%) !important; }
    .login-box {
        background: white; border-radius: 24px; padding: 2.5rem 2rem;
        max-width: 420px; margin: 6vh auto 0 auto;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        text-align: center;
    }
    .login-logo-row {
        display: flex; justify-content: center; gap: 14px; margin-bottom: 1.5rem;
    }
    .login-logo-pt {
        background: linear-gradient(135deg,#00D2AA,#00A88A);
        border-radius: 14px; padding: 12px 18px;
        font-size: 1.2rem; font-weight: 700; color: white;
    }
    .login-logo-cob {
        background: #0F3460;
        border-radius: 14px; padding: 12px 18px;
        font-size: 1.2rem; font-weight: 700; color: white;
    }
    .login-title {
        font-family: 'DM Serif Display', serif !important;
        color: #0D2137 !important; font-size: 1.6rem !important;
        margin-bottom: 0.3rem !important;
    }
    .login-subtitle { color: #6B7F94 !important; font-size: 0.88rem !important; margin-bottom: 1.5rem !important; }
    .login-error {
        background: #FFF0F0; border: 1px solid #FFCCCC;
        border-radius: 10px; padding: 10px 14px;
        color: #CC3333 !important; font-size: 0.85rem; margin-top: 0.8rem;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 12px !important;
        border: 2px solid #E8EEF4 !important;
        padding: 0.7rem 1rem !important;
        font-size: 0.95rem !important;
        text-align: center !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #00D2AA !important;
        box-shadow: 0 0 0 3px rgba(0,210,170,0.15) !important;
    }
    div[data-testid="stForm"] > div { background: transparent !important; border: none !important; }
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #00D2AA, #00A88A) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; width: 100% !important;
        padding: 0.7rem !important; font-size: 1rem !important;
        font-weight: 600 !important; letter-spacing: 0.3px !important;
        margin-top: 0.5rem !important;
        transition: opacity 0.2s !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover { opacity: 0.88 !important; }
    </style>
    """, unsafe_allow_html=True)
 
    st.markdown("""
    <div class="login-box">
        <div class="login-logo-row">
            <div class="login-logo-pt">PT<br><span style="font-size:0.5rem;letter-spacing:1px;font-weight:400">OF THE CITY</span></div>
            <div class="login-logo-cob">COB<br><span style="font-size:0.5rem;letter-spacing:1px;font-weight:400">SOLUTION</span></div>
        </div>
        <p class="login-title">Quality Assistant</p>
        <p class="login-subtitle">Enter your access password to continue</p>
    </div>
    """, unsafe_allow_html=True)
 
    # center the form using columns
    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form("login_form"):
            password = st.text_input("", placeholder="Enter password…", type="password", label_visibility="collapsed")
            submitted = st.form_submit_button("Access System →")
            if submitted:
                if password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.markdown('<div class="login-error">❌ Incorrect password. Please try again.</div>',
                                unsafe_allow_html=True)
    return False
 
 
# ══════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
    #MainMenu, footer, header, .stDeployButton,
    [data-testid="stToolbar"] { display: none !important; }
    .stApp { background: #F0F4F8 !important; }
 
    [data-testid="stSidebar"] {
        background: linear-gradient(170deg, #0D2137 0%, #0F3460 60%, #0E4D7B 100%) !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] * { color: #E0EAF4 !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #E0EAF4 !important; border-radius: 10px !important; width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0,210,170,0.2) !important; border-color: #00D2AA !important;
    }
 
    .main .block-container { padding: 0 2rem 2rem 2rem !important; max-width: 900px !important; }
 
    .chat-header {
        background: linear-gradient(90deg, #0D2137 0%, #0F3460 100%);
        padding: 1.2rem 2rem; border-radius: 0 0 20px 20px;
        display: flex; align-items: center; gap: 16px;
        margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(13,33,55,0.15);
    }
    .chat-header-logo {
        width: 42px; height: 42px;
        background: linear-gradient(135deg, #00D2AA, #00A88A);
        border-radius: 12px; display: flex; align-items: center;
        justify-content: center; font-size: 20px;
    }
    .chat-header-text h1 {
        font-family: 'DM Serif Display', serif !important;
        color: white !important; font-size: 1.3rem !important;
        margin: 0 !important; padding: 0 !important; line-height: 1.2 !important;
    }
    .chat-header-text p {
        color: #00D2AA !important; font-size: 0.75rem !important;
        margin: 0 !important; font-weight: 500; letter-spacing: 0.5px;
    }
    .chat-header-badge {
        margin-left: auto;
        background: rgba(0,210,170,0.15); border: 1px solid rgba(0,210,170,0.3);
        color: #00D2AA !important; padding: 4px 12px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.5px;
    }
 
    [data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0.3rem 0 !important; }
 
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
        background: linear-gradient(135deg, #0F3460, #0D2137) !important;
        color: white !important; border-radius: 18px 18px 4px 18px !important;
        padding: 0.9rem 1.2rem !important; max-width: 75% !important;
        margin-left: auto !important; box-shadow: 0 2px 12px rgba(13,33,55,0.15) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
        background: white !important; color: #1a2a3a !important;
        border-radius: 18px 18px 18px 4px !important; padding: 0.9rem 1.2rem !important;
        max-width: 82% !important; box-shadow: 0 2px 12px rgba(13,33,55,0.08) !important;
        border: 1px solid #E8EEF4 !important;
    }
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #00D2AA, #00A88A) !important;
        color: white !important; border-radius: 50% !important;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #0F3460, #0D2137) !important;
        color: white !important; border-radius: 50% !important;
    }
 
    [data-testid="stChatInput"] {
        background: white !important; border: 2px solid #E8EEF4 !important;
        border-radius: 16px !important; box-shadow: 0 4px 20px rgba(13,33,55,0.08) !important;
    }
    [data-testid="stChatInput"]:focus-within { border-color: #00D2AA !important; }
    [data-testid="stChatInput"] textarea { font-family: 'DM Sans', sans-serif !important; color: #1a2a3a !important; }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #00D2AA, #00A88A) !important;
        border-radius: 10px !important; color: white !important;
    }
    .stSpinner > div { border-top-color: #00D2AA !important; }
 
    .question-chip {
        background: rgba(0,210,170,0.08); border: 1px solid rgba(0,210,170,0.2);
        border-radius: 8px; padding: 7px 11px; margin: 4px 0;
        font-size: 0.78rem; color: #c8dff0 !important; display: block;
    }
 
    .welcome-card {
        background: white; border-radius: 20px; padding: 2rem; text-align: center;
        border: 1px solid #E8EEF4; box-shadow: 0 4px 20px rgba(13,33,55,0.06);
        margin: 1rem 0 2rem 0;
    }
    .welcome-card h2 { font-family: 'DM Serif Display', serif !important; color: #0D2137 !important; font-size: 1.5rem !important; }
    .welcome-card p { color: #6B7F94 !important; font-size: 0.9rem !important; }
    .welcome-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 1.2rem; }
    .welcome-tile {
        background: #F7FAFC; border: 1px solid #E8EEF4; border-radius: 12px;
        padding: 12px 14px; text-align: left; font-size: 0.82rem; color: #3D5A73 !important;
    }
    .welcome-tile span { display: block; font-size: 1.2rem; margin-bottom: 4px; }
 
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-thumb { background: #CBD5E0; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════
# DATA & INDEX
# ══════════════════════════════════════════════
def load_text(text_json, ocr_json):
    pages = {}
    if os.path.exists(text_json):
        try:
            with open(text_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, content in data.items():
                    pages[key.replace("page_", "")] = content.strip()
        except Exception as e:
            st.warning(f"Could not load {text_json}: {e}")
    
    if os.path.exists(ocr_json):
        try:
            with open(ocr_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, content in data.items():
                    pn = key.replace("page_", "")
                    if len(content.strip()) > len(pages.get(pn, "")):
                        pages[pn] = content.strip()
        except Exception as e:
            st.warning(f"Could not load {ocr_json}: {e}")
    
    return [{"page": k, "text": v}
            for k, v in sorted(pages.items(), key=lambda x: int(x[0])) if v]
 
def chunk_pages(pages, size, overlap):
    chunks = []
    for p in pages:
        t, pg, s = p["text"], p["page"], 0
        while s < len(t):
            chunk = t[s:s+size]
            if chunk.strip():
                chunks.append({"page": pg, "text": chunk})
            s += size - overlap
    return chunks
 
@st.cache_resource(show_spinner=False)
def load_index():
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "rb") as f:
                saved = pickle.load(f)
            return saved["index"], saved["chunks"], saved["model_name"]
        except Exception as e:
            st.warning(f"Could not load cached index: {e}. Rebuilding...")
    
    pages  = load_text(TEXT_JSON, OCR_JSON)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Error handling for empty data
    if not chunks:
        error_msg = f"""
        ⚠️ **No text chunks found!** 
        
        Please check:
        1. **{TEXT_JSON}** - {'✓ exists' if os.path.exists(TEXT_JSON) else '✗ missing'}
        2. **{OCR_JSON}** - {'✓ exists' if os.path.exists(OCR_JSON) else '✗ missing'}
        
        Current DATA_DIR: `{DATA_DIR}`
        
        Make sure your JSON files contain data in the format:
        ```json
        {{
            "page_1": "text content here",
            "page_2": "more content"
        }}
        ```
        """
        st.error(error_msg)
        st.stop()
    
    model_name = "all-MiniLM-L6-v2"
    embedder   = SentenceTransformer(model_name)
    embs = np.array(embedder.encode([c["text"] for c in chunks], batch_size=64)).astype("float32")
    
    # Only normalize if we have valid embeddings
    if embs.size > 0:
        faiss.normalize_L2(embs)
    
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    
    # Try to save the index
    try:
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"index": index, "chunks": chunks, "model_name": model_name}, f)
    except Exception as e:
        st.warning(f"Could not save index to disk: {e}")
    
    return index, chunks, model_name
 
@st.cache_resource(show_spinner=False)
def load_embedder(name):
    return SentenceTransformer(name)
 
def retrieve(query, index, chunks, embedder, top_k):
    q = np.array(embedder.encode([query])).astype("float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], idxs[0]) if i < len(chunks)]
 
def ask_groq(question, context_chunks, history):
    client  = Groq(api_key=GROQ_API_KEY)
    context = "\n\n---\n\n".join([f"[Page {c['page']}]\n{c['text']}" for c in context_chunks])
    system_prompt = f"""You are a professional Quality Procedures assistant for PT of the City,
powered by the COB Solution knowledge base.
 
Your role is to help staff and management understand Quality Procedures clearly and confidently.
 
How to respond:
1. Start by briefly explaining the concept in plain language (2-3 sentences max)
2. Then provide the specific details found in the document
3. Add practical context: "In practice, this means..." when helpful
4. End with a natural follow-up offer if relevant
 
Style rules:
- Warm, professional, and clear — like a knowledgeable colleague
- Use numbered lists for steps or procedures
- Use tables for comparisons
- Always cite page numbers when referencing specific content
- If something is not in the document, say "Based on general practice..." and answer from knowledge
 
DOCUMENT CONTEXT:
{context}
"""
    messages = [{"role": "system", "content": system_prompt}]
    for t in history[-6:]:
        messages += [{"role": "user", "content": t["user"]},
                     {"role": "assistant", "content": t["assistant"]}]
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.2, max_tokens=1500
    )
    return resp.choices[0].message.content
 
 
# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="PT of the City — Quality Assistant",
        page_icon="🏥", layout="wide",
        initial_sidebar_state="expanded",
    )
 
    # ── PASSWORD GATE ──
    if not check_password():
        st.stop()
 
    inject_css()
 
    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#00D2AA,#00A88A);
                        border-radius:12px;padding:10px;text-align:center;margin-bottom:6px">
                <div style="font-size:1.1rem;font-weight:700;color:white">PT</div>
                <div style="font-size:0.55rem;color:rgba(255,255,255,0.85);letter-spacing:1px">OF THE CITY</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);
                        border-radius:12px;padding:10px;text-align:center;margin-bottom:6px">
                <div style="font-size:1.1rem;font-weight:700;color:white">COB</div>
                <div style="font-size:0.55rem;color:rgba(255,255,255,0.6);letter-spacing:1px">SOLUTION</div>
            </div>""", unsafe_allow_html=True)
 
        st.markdown("<div style='height:1px;background:rgba(255,255,255,0.1);margin:14px 0'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.7rem;letter-spacing:1.5px;font-weight:600;color:rgba(255,255,255,0.4);margin-bottom:10px'>QUICK QUESTIONS</p>", unsafe_allow_html=True)
 
        for icon, q in [
            ("📋", "What are the onboarding steps?"),
            ("📊", "Show all SLAs in the document"),
            ("🔍", "What are the KPIs and targets?"),
            ("⚖️", "Compare escalation levels"),
            ("📝", "Summarize document approval process"),
            ("🎯", "What are quality control standards?"),
        ]:
            st.markdown(f'<div class="question-chip">{icon} {q}</div>', unsafe_allow_html=True)
 
        st.markdown("<div style='height:1px;background:rgba(255,255,255,0.1);margin:14px 0'></div>", unsafe_allow_html=True)
 
        if st.button("🗑️  Clear conversation"):
            st.session_state.history = []
            st.rerun()
 
        if st.button("🔒  Log out"):
            st.session_state.authenticated = False
            st.session_state.history = []
            st.rerun()
 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.7rem;color:rgba(255,255,255,0.3);text-align:center'>Powered by Groq · Llama 3 · FAISS</p>", unsafe_allow_html=True)
 
    # ── LOAD INDEX ──
    with st.spinner("Initializing knowledge base…"):
        index, chunks, model_name = load_index()
        embedder = load_embedder(model_name)
 
    # Show data status in sidebar
    with st.sidebar:
        st.markdown("<div style='height:1px;background:rgba(255,255,255,0.1);margin:14px 0'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <p style='font-size:0.65rem;color:rgba(255,255,255,0.3);text-align:center'>
        📊 {len(chunks)} chunks loaded<br>
        📁 Data: {os.path.basename(DATA_DIR)}
        </p>
        """, unsafe_allow_html=True)
 
    # ── HEADER ──
    st.markdown("""
    <div class="chat-header">
        <div class="chat-header-logo">🏥</div>
        <div class="chat-header-text">
            <h1>PT of the City — Quality Assistant</h1>
            <p>COB SOLUTION KNOWLEDGE BASE</p>
        </div>
        <div class="chat-header-badge">● ONLINE</div>
    </div>
    """, unsafe_allow_html=True)
 
    if "history" not in st.session_state:
        st.session_state.history = []
 
    if not st.session_state.history:
        st.markdown("""
        <div class="welcome-card">
            <h2>How can I help you today?</h2>
            <p>I have full knowledge of the Quality Procedures document.<br>
               Ask me anything about procedures, SLAs, KPIs, or policies.</p>
            <div class="welcome-grid">
                <div class="welcome-tile"><span>📋</span>Find procedures & steps</div>
                <div class="welcome-tile"><span>📊</span>Extract data & numbers</div>
                <div class="welcome-tile"><span>⚖️</span>Compare procedures</div>
                <div class="welcome-tile"><span>📝</span>Summarize any topic</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["assistant"])
 
    question = st.chat_input("Ask about any procedure, SLA, policy, or standard…")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching & generating answer…"):
                relevant = retrieve(question, index, chunks, embedder, TOP_K)
                answer   = ask_groq(question, relevant, st.session_state.history)
            st.write(answer)
        st.session_state.history.append({"user": question, "assistant": answer, "sources": relevant})
        st.rerun()
 
 
if __name__ == "__main__":
    main()