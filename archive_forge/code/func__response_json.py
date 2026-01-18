import base64
from typing import Dict, Optional
from urllib.parse import quote
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _response_json(self, url: str) -> dict:
    """Use requests to run request to DataForSEO SERP API and return results."""
    request_details = self._prepare_request(url)
    response = requests.post(request_details['url'], headers=request_details['headers'], json=request_details['data'])
    response.raise_for_status()
    return self._check_response(response.json())