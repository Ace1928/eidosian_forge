import warnings
from typing import Any, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
class DDGInput(BaseModel):
    """Input for the DuckDuckGo search tool."""
    query: str = Field(description='search query to look up')