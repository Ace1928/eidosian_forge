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
def _document_details_for_docset_id(self, docset_id: str) -> List[Dict]:
    """Gets all document details for the given docset ID"""
    url = f'{self.api}/docsets/{docset_id}/documents'
    all_documents = []
    while url:
        response = requests.get(url, headers={'Authorization': f'Bearer {self.access_token}'})
        if response.ok:
            data = response.json()
            all_documents.extend(data['documents'])
            url = data.get('next', None)
        else:
            raise Exception(f'Failed to download {url} (status: {response.status_code})')
    return all_documents