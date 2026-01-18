from __future__ import annotations
import hashlib
import json
import uuid
from itertools import islice
from typing import (
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.vectorstores import VectorStore
from langchain.indexes.base import NAMESPACE_UUID, RecordManager
def _get_source_id_assigner(source_id_key: Union[str, Callable[[Document], str], None]) -> Callable[[Document], Union[str, None]]:
    """Get the source id from the document."""
    if source_id_key is None:
        return lambda doc: None
    elif isinstance(source_id_key, str):
        return lambda doc: doc.metadata[source_id_key]
    elif callable(source_id_key):
        return source_id_key
    else:
        raise ValueError(f'source_id_key should be either None, a string or a callable. Got {source_id_key} of type {type(source_id_key)}.')