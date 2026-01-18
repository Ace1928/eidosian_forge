import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from transformers import BertModel, BertTokenizer
import torch

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


def generate_embeddings(text):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use the CLS token's embedding


def text_analysis(text):
    tokens = word_tokenize(text)
    stopwords_removed = [
        word for word in tokens if word not in stopwords.words("english")
    ]
    pos_tags = pos_tag(stopwords_removed)
    return {
        "tokens": tokens,
        "stopwords_removed": stopwords_removed,
        "pos_tags": pos_tags,
    }


class EmbeddingManager:
    def __init__(self, connection):
        self.con = connection

    def insert_document(self, doc_id, text):
        analysis = text_analysis(text)
        embedding = generate_embeddings(text)
        self.con.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?)",
            (doc_id, text, embedding.tobytes(), analysis),
        )

    def search_documents(self, query):
        query_embedding = generate_embeddings(query)
        query_embedding_bytes = query_embedding.tobytes()

        result = self.con.execute(
            """
        SELECT id, text, cosine_similarity(embedding, ?) AS similarity
        FROM documents
        ORDER BY similarity DESC
        LIMIT 10
        """,
            (query_embedding_bytes,),
        ).fetchdf()
        return result


import streamlit as st

st.title("Semantic Search with DuckDB")

# Initialize the database and manager
con = duckdb.connect(database=":memory:", read_only=False)
manager = EmbeddingManager(con)
con.execute(
    "CREATE TABLE documents (id INTEGER, text VARCHAR, embedding BLOB, metadata JSON)"
)

# Interface to add documents
with st.form("add_document"):
    doc_id = st.number_input("Document ID", min_value=1, value=1, step=1)
    text = st.text_area("Document Text")
    submitted = st.form_submit_button("Insert Document")
    if submitted:
        manager.insert_document(doc_id, text)
        st.success("Document inserted successfully!")

# Interface to perform search
with st.form("search"):
    query = st.text_input("Search Query")
    search_submitted = st.form_submit_button("Search")
    if search_submitted:
        results = manager.search_documents(query)
        st.write(results)

st.button("Re-run")

if __name__ == "__main__":
    st.run()
