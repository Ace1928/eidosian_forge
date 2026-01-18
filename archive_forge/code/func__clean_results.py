import json
from typing import Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def _clean_results(self, raw_search_results: List[Dict]) -> List[Dict]:
    cleaned_results = []
    for result in raw_search_results:
        cleaned_results.append({'title': result.get('title', 'Unknown Title'), 'url': result.get('url', 'Unknown URL'), 'author': result.get('author', 'Unknown Author'), 'published_date': result.get('publishedDate', 'Unknown Date')})
    return cleaned_results