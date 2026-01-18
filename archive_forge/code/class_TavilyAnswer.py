from typing import Dict, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
class TavilyAnswer(BaseTool):
    """Tool that queries the Tavily Search API and gets back an answer."""
    name: str = 'tavily_answer'
    description: str = 'A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query. This returns only the answer - not the original source data.'
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)
    args_schema: Type[BaseModel] = TavilyInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.raw_results(query, max_results=5, include_answer=True, search_depth='basic')['answer']
        except Exception as e:
            return repr(e)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            result = await self.api_wrapper.raw_results_async(query, max_results=5, include_answer=True, search_depth='basic')
            return result['answer']
        except Exception as e:
            return repr(e)