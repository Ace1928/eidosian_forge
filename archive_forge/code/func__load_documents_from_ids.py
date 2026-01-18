import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_community.document_loaders.base import BaseLoader
def _load_documents_from_ids(self) -> List[Document]:
    """Load documents from a list of IDs."""
    if not self.document_ids:
        raise ValueError('document_ids must be set')
    return [self._load_document_from_id(doc_id) for doc_id in self.document_ids]