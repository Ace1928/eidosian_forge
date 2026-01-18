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
def _index_schema(self) -> Optional[dict]:
    """Return the index schema as a dictionary.
        Return None if no schema found.
        """
    if self._is_direct_access_index():
        schema_json = self._direct_access_index_spec.get('schema_json')
        if schema_json is not None:
            return json.loads(schema_json)
    return None