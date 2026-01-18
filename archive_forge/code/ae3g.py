import os
import torch
import logging
import streamlit as st
import duckdb
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration

logging.basicConfig(level=logging.INFO)

model_name = "microsoft/Phi-3-mini-128k-instruct"
model_path = (
    "/home/lloyd/Downloads/local_model_store/microsoft/Phi-3-mini-128k-instruct"
)
db_path = "/home/lloyd/Downloads/local_model_store/conversation_embeddings.db"
embedding_model_path = "/home/lloyd/Downloads/local_model_store/t5-small"

# Initialize database connection
conn = duckdb.connect(database=db_path, read_only=False)
conn.execute(
    "CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, input TEXT, output TEXT, input_embedding BLOB, output_embedding BLOB)"
)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
embedding_model = T5ForConditionalGeneration.from_pretrained(embedding_model_path)
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)

# Set device based on your hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
embedding_model.to(device)
logging.info(f"Models moved to device: {device}")

# Streamlit app setup
st.set_page_config(page_title="Interactive Text Generation with Phi-3", layout="wide")
st.title("Interactive Text Generation with Phi-3")

# Create two columns for chat interface
col1, col2 = st.columns([2, 1])

# Conversation history
conversation_history = conn.execute("SELECT input, output FROM embeddings").fetchall()

with col2:
    st.header("Conversation Log")
    conversation_container = st.empty()

# Display conversation history
with conversation_container.container():
    for input_text, output_text in conversation_history:
        st.markdown(f"**User:** {input_text}")
        st.markdown(f"**Assistant:** {output_text}")
        st.markdown("---")


# Text generation function
@st.cache_resource
def generate_text(prompt: str):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    logging.info("Input encoded and moved to device.")
    try:
        outputs = model.generate(
            inputs,
            max_new_tokens=500,
            top_p=0.95,
            do_sample=True,
            top_k=60,
            temperature=0.95,
            early_stopping=True,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info("Text generation successful.")
        return generated_text
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        return "Error in text generation."


# Embedding generation function
@st.cache_resource
def generate_embeddings(text: str, tokenizer, model):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = (
            model.encoder(input_ids).last_hidden_state.mean(dim=1).cpu().numpy()
        )
    return embeddings


# Text reconstruction from embeddings function
@st.cache_resource
def reconstruct_text(embeddings, tokenizer, model):
    embeddings_tensor = torch.tensor(embeddings).unsqueeze(0).to(device)
    output = model.generate(inputs_embeds=embeddings_tensor)
    reconstructed_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return reconstructed_text


# Streamlit interaction
with col1:
    user_input = st.text_input("Enter your prompt:", key="user_input")
    if user_input:
        generated_text = generate_text(user_input)

        # Display generated text
        st.markdown(f"**Assistant:** {generated_text}")

        # Embedding and storing the conversation
        input_embeddings = generate_embeddings(
            user_input, embedding_tokenizer, embedding_model
        )
        output_embeddings = generate_embeddings(
            generated_text, embedding_tokenizer, embedding_model
        )

        conn.execute(
            "INSERT INTO embeddings (input, output, input_embedding, output_embedding) VALUES (?, ?, ?, ?)",
            [
                user_input,
                generated_text,
                input_embeddings.tobytes(),
                output_embeddings.tobytes(),
            ],
        )
        logging.info("Conversation embeddings stored.")

        # Update conversation history
        conversation_container.empty()
        with conversation_container.container():
            for input_text, output_text in conn.execute(
                "SELECT input, output FROM embeddings"
            ).fetchall():
                st.markdown(f"**User:** {input_text}")
                st.markdown(f"**Assistant:** {output_text}")
                st.markdown("---")
