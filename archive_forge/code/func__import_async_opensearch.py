from __future__ import annotations
import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _import_async_opensearch() -> Any:
    """Import AsyncOpenSearch if available, otherwise raise error."""
    try:
        from opensearchpy import AsyncOpenSearch
    except ImportError:
        raise ImportError(IMPORT_ASYNC_OPENSEARCH_PY_ERROR)
    return AsyncOpenSearch