from typing import Dict, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
class TavilyInput(BaseModel):
    """Input for the Tavily tool."""
    query: str = Field(description='search query to look up')