from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _check_scope_and_collection_exists(self) -> bool:
    """Check if the scope and collection exists in the linked Couchbase bucket
        Raises a ValueError if either is not found"""
    scope_collection_map: Dict[str, Any] = {}
    for scope in self._bucket.collections().get_all_scopes():
        scope_collection_map[scope.name] = []
        for collection in scope.collections:
            scope_collection_map[scope.name].append(collection.name)
    if self._scope_name not in scope_collection_map.keys():
        raise ValueError(f'Scope {self._scope_name} not found in Couchbase bucket {self._bucket_name}')
    if self._collection_name not in scope_collection_map[self._scope_name]:
        raise ValueError(f'Collection {self._collection_name} not found in scope {self._scope_name} in Couchbase bucket {self._bucket_name}')
    return True