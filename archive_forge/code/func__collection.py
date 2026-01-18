from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
@property
def _collection(self) -> Collection:
    return self._typesense_client.collections[self._typesense_collection_name]