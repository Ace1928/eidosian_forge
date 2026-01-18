from __future__ import annotations
import logging
import os
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Union
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore
from packaging import version
from langchain_community.vectorstores.utils import (
def _import_pinecone() -> Any:
    try:
        import pinecone
    except ImportError as e:
        raise ImportError('Could not import pinecone python package. Please install it with `pip install pinecone-client`.') from e
    return pinecone