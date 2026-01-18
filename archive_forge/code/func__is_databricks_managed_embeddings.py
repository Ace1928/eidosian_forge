from __future__ import annotations
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _is_databricks_managed_embeddings(self) -> bool:
    """Return True if the embeddings are managed by Databricks Vector Search."""
    return self._is_delta_sync_index() and self._embedding_source_column_name() is not None