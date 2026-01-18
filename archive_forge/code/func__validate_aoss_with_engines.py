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
def _validate_aoss_with_engines(is_aoss: bool, engine: str) -> None:
    """Validate AOSS with the engine."""
    if is_aoss and engine != 'nmslib' and (engine != 'faiss'):
        raise ValueError('Amazon OpenSearch Service Serverless only supports `nmslib` or `faiss` engines')