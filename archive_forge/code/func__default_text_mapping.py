from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def _default_text_mapping(dim: int) -> Dict:
    return {'properties': {'text': {'type': 'text'}, 'vector': {'type': 'dense_vector', 'dims': dim}}}