from typing import Dict, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
def _ddgs_news(self, query: str, max_results: Optional[int]=None) -> List[Dict[str, str]]:
    """Run query through DuckDuckGo news search and return results."""
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        ddgs_gen = ddgs.news(query, region=self.region, safesearch=self.safesearch, timelimit=self.time, max_results=max_results or self.max_results)
        if ddgs_gen:
            return [r for r in ddgs_gen]
    return []