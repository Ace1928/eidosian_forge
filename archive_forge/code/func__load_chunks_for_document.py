import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_community.document_loaders.base import BaseLoader
def _load_chunks_for_document(self, document_id: str, docset_id: str, document_name: Optional[str]=None, additional_metadata: Optional[Mapping]=None) -> List[Document]:
    """Load chunks for a document."""
    url = f'{self.api}/docsets/{docset_id}/documents/{document_id}/dgml'
    response = requests.request('GET', url, headers={'Authorization': f'Bearer {self.access_token}'}, data={})
    if response.ok:
        return self._parse_dgml(content=response.content, document_name=document_name, additional_doc_metadata=additional_metadata)
    else:
        raise Exception(f'Failed to download {url} (status: {response.status_code})')