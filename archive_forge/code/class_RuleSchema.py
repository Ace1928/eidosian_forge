import builtins
import json
from typing import Optional, Type
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.ainetwork.base import AINBaseTool, OperationType
class RuleSchema(BaseModel):
    """Schema for owner operations."""
    type: OperationType = Field(...)
    path: str = Field(..., description='Path on the blockchain where the rule applies')
    eval: Optional[str] = Field(None, description='eval string to determine permission')