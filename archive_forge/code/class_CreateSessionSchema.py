from typing import TYPE_CHECKING, Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
class CreateSessionSchema(BaseModel):
    """Input for CreateSessionTool."""
    query: str = Field(..., description='The query to run in multion agent.')
    url: str = Field('https://www.google.com/', description='The Url to run the agent at. Note: accepts only secure             links having https://')