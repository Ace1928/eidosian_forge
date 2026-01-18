from pathlib import Path
from typing import Dict, Iterator, List, Optional
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_community.document_loaders.base import BaseLoader
def _get_page_content(self, page_id: str) -> str:
    """Get page content from OneNote API"""
    request_url = self.onenote_api_base_url + f'/pages/{page_id}/content'
    response = requests.get(request_url, headers=self._headers, timeout=10)
    response.raise_for_status()
    return response.text