import base64
from typing import Dict, Optional
from urllib.parse import quote
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _check_response(self, response: dict) -> dict:
    """Check the response from the DataForSEO SERP API for errors."""
    if response.get('status_code') != 20000:
        raise ValueError(f'Got error from DataForSEO SERP API: {response.get('status_message')}')
    return response