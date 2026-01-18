import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
def _document_get_date(self, field: str, document: Document) -> datetime.datetime:
    """Return the value of the date field of a document."""
    if field in document.metadata:
        if isinstance(document.metadata[field], float):
            return datetime.datetime.fromtimestamp(document.metadata[field])
        return document.metadata[field]
    return datetime.datetime.now()