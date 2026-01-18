import warnings
from typing import Any, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
class DuckDuckGoSearchResults(BaseTool):
    """Tool that queries the DuckDuckGo search API and gets back json."""
    name: str = 'duckduckgo_results_json'
    description: str = 'A wrapper around Duck Duck Go Search. Useful for when you need to answer questions about current events. Input should be a search query. Output is a JSON array of the query results'
    max_results: int = Field(alias='num_results', default=4)
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(default_factory=DuckDuckGoSearchAPIWrapper)
    backend: str = 'text'
    args_schema: Type[BaseModel] = DDGInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        res = self.api_wrapper.results(query, self.max_results, source=self.backend)
        res_strs = [', '.join([f'{k}: {v}' for k, v in d.items()]) for d in res]
        return ', '.join([f'[{rs}]' for rs in res_strs])