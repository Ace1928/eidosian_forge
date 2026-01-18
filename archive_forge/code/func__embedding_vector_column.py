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
def _embedding_vector_column(self) -> dict:
    """Return the embedding vector column configs as a dictionary.
        Empty if the index is not a self-managed embedding index.
        """
    index_spec = self._delta_sync_index_spec if self._is_delta_sync_index() else self._direct_access_index_spec
    return next(iter(index_spec.get('embedding_vector_columns') or list()), dict())