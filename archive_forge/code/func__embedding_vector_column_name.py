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
def _embedding_vector_column_name(self) -> Optional[str]:
    """Return the name of the embedding vector column.
        None if the index is not a self-managed embedding index.
        """
    return self._embedding_vector_column().get('name')