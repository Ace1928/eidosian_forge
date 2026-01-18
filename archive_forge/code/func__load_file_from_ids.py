import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_community.document_loaders.base import BaseLoader
def _load_file_from_ids(self) -> List[Document]:
    """Load files from a list of IDs."""
    if not self.file_ids:
        raise ValueError('file_ids must be set')
    docs = []
    for file_id in self.file_ids:
        docs.extend(self._load_file_from_id(file_id))
    return docs