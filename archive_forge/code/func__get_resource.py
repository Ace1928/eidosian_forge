import json
import urllib.request
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.utils import get_from_env, stringify_dict
from langchain_community.document_loaders.base import BaseLoader
def _get_resource(self) -> List[Document]:
    endpoint = IUGU_ENDPOINTS.get(self.resource)
    if endpoint is None:
        return []
    return self._make_request(endpoint)