from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.arxiv import ArxivAPIWrapper
class ArxivInput(BaseModel):
    """Input for the Arxiv tool."""
    query: str = Field(description='search query to look up')