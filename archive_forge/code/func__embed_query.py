from __future__ import annotations
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def _embed_query(self, text: str) -> List[float]:
    if isinstance(self._embedding, Embeddings):
        return self._embedding.embed_query(text)
    return self._embedding(text)