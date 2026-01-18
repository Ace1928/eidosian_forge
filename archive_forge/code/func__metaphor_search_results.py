import json
from typing import Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def _metaphor_search_results(self, query: str, num_results: int, include_domains: Optional[List[str]]=None, exclude_domains: Optional[List[str]]=None, start_crawl_date: Optional[str]=None, end_crawl_date: Optional[str]=None, start_published_date: Optional[str]=None, end_published_date: Optional[str]=None, use_autoprompt: Optional[bool]=None) -> List[dict]:
    headers = {'X-Api-Key': self.metaphor_api_key}
    params = {'numResults': num_results, 'query': query, 'includeDomains': include_domains, 'excludeDomains': exclude_domains, 'startCrawlDate': start_crawl_date, 'endCrawlDate': end_crawl_date, 'startPublishedDate': start_published_date, 'endPublishedDate': end_published_date, 'useAutoprompt': use_autoprompt}
    response = requests.post(f'{METAPHOR_API_URL}/search', headers=headers, json=params)
    response.raise_for_status()
    search_results = response.json()
    return search_results['results']