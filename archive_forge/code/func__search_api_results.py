from typing import Any, Dict, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
def _search_api_results(self, query: str, **kwargs: Any) -> dict:
    request_details = self._prepare_request(query, **kwargs)
    response = requests.get(url=request_details['url'], params=request_details['params'], headers=request_details['headers'])
    response.raise_for_status()
    return response.json()