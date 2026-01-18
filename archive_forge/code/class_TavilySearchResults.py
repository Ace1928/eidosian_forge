from typing import Dict, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
class TavilySearchResults(BaseTool):
    """Tool that queries the Tavily Search API and gets back json."""
    name: str = 'tavily_search_results_json'
    description: str = 'A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.'
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)
    max_results: int = 5
    args_schema: Type[BaseModel] = TavilyInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.results(query, self.max_results)
        except Exception as e:
            return repr(e)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            return await self.api_wrapper.results_async(query, self.max_results)
        except Exception as e:
            return repr(e)