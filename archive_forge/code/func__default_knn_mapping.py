from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
@staticmethod
def _default_knn_mapping(dims: int, similarity: Optional[str]='dot_product') -> Dict:
    return {'properties': {'text': {'type': 'text'}, 'vector': {'type': 'dense_vector', 'dims': dims, 'index': True, 'similarity': similarity}}}