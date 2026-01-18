import builtins
import json
from enum import Enum
from typing import List, Optional, Type, Union
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.ainetwork.base import AINBaseTool
class AppSchema(BaseModel):
    """Schema for app operations."""
    type: AppOperationType = Field(...)
    appName: str = Field(..., description='Name of the application on the blockchain')
    address: Optional[Union[str, List[str]]] = Field(None, description="A single address or a list of addresses. Default: current session's address")