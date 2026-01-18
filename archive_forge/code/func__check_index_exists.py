from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _check_index_exists(self) -> bool:
    """Check if the Search index exists in the linked Couchbase cluster
        Raises a ValueError if the index does not exist"""
    if self._scoped_index:
        all_indexes = [index.name for index in self._scope.search_indexes().get_all_indexes()]
        if self._index_name not in all_indexes:
            raise ValueError(f'Index {self._index_name} does not exist.  Please create the index before searching.')
    else:
        all_indexes = [index.name for index in self._cluster.search_indexes().get_all_indexes()]
        if self._index_name not in all_indexes:
            raise ValueError(f'Index {self._index_name} does not exist.  Please create the index before searching.')
    return True