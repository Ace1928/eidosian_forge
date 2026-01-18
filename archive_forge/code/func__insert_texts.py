from __future__ import annotations
import logging
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _insert_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List:
    if not texts:
        return []
    embeddings = self._embedding.embed_documents(texts)
    to_insert = [{self._text_key: t, self._embedding_key: embedding, **m} for t, m, embedding in zip(texts, metadatas, embeddings)]
    insert_result = self._collection.insert_many(to_insert)
    return insert_result.inserted_ids