import logging
from typing import Any, Dict, List, Optional
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
def _outline_api_query(self, query: str) -> List:
    raw_result = requests.post(f'{self.outline_instance_url}{self.outline_search_endpoint}', data={'query': query, 'limit': self.top_k_results}, headers={'Authorization': f'Bearer {self.outline_api_key}'})
    if not raw_result.ok:
        raise ValueError('Outline API returned an error: ', raw_result.text)
    return raw_result.json()['data']