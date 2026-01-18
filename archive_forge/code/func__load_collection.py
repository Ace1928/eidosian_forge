from __future__ import annotations
import logging
import warnings
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _load_collection(self) -> DocumentCollection:
    """
        Load the collection from the Zep backend.
        """
    from zep_python import NotFoundError
    try:
        collection = self._client.document.get_collection(self.collection_name)
    except NotFoundError:
        logger.info(f'Collection {self.collection_name} not found. Creating new collection.')
        collection = self._create_collection()
    return collection