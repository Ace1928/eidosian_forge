from enum import Enum
from typing import Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.vectorstores import VectorStore
from langchain.storage._lc_store import create_kv_docstore
class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""
    similarity = 'similarity'
    'Similarity search.'
    mmr = 'mmr'
    'Maximal Marginal Relevance reranking of similarity search.'