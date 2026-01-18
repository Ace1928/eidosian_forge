import json
from typing import List
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
def _search_request(self, query: str) -> List[dict]:
    headers = {'X-Subscription-Token': self.api_key, 'Accept': 'application/json'}
    req = requests.PreparedRequest()
    params = {**self.search_kwargs, **{'q': query}}
    req.prepare_url(self.base_url, params)
    if req.url is None:
        raise ValueError('prepared url is None, this should not happen')
    response = requests.get(req.url, headers=headers)
    if not response.ok:
        raise Exception(f'HTTP error {response.status_code}')
    return response.json().get('web', {}).get('results', [])