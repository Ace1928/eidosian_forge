import json
import urllib.request
from typing import Any, Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_larksuite_api_json_data(self, api_url: str) -> Any:
    """Get LarkSuite (FeiShu) API response json data."""
    headers = {'Authorization': f'Bearer {self.access_token}'}
    request = urllib.request.Request(api_url, headers=headers)
    with urllib.request.urlopen(request) as response:
        json_data = json.loads(response.read().decode())
        return json_data