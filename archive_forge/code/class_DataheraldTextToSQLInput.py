from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.dataherald import DataheraldAPIWrapper
class DataheraldTextToSQLInput(BaseModel):
    prompt: str = Field(description='Natural language query to be translated to a SQL query.')