from typing import Dict, List
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _bing_search_results(self, search_term: str, count: int) -> List[dict]:
    headers = {'Ocp-Apim-Subscription-Key': self.bing_subscription_key}
    params = {'q': search_term, 'count': count, 'textDecorations': True, 'textFormat': 'HTML', **self.search_kwargs}
    response = requests.get(self.bing_search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    if 'webPages' in search_results:
        return search_results['webPages']['value']
    return []