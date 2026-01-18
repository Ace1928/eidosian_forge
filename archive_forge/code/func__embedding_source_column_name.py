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
def _embedding_source_column_name(self) -> Optional[str]:
    """Return the name of the embedding source column.
        None if the index is not a Databricks-managed embedding index.
        """
    return self._embedding_source_column().get('name')