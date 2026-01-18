from typing import Optional
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.dataforseo_api_search import DataForSeoAPIWrapper
class DataForSeoAPISearchRun(BaseTool):
    """Tool that queries the DataForSeo Google search API."""
    name: str = 'dataforseo_api_search'
    description: str = 'A robust Google Search API provided by DataForSeo.This tool is handy when you need information about trending topics or current events.'
    api_wrapper: DataForSeoAPIWrapper

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        return str(self.api_wrapper.run(query))

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        """Use the tool asynchronously."""
        return (await self.api_wrapper.arun(query)).__str__()