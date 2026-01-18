from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.tools.file_management.utils import (
class WriteFileInput(BaseModel):
    """Input for WriteFileTool."""
    file_path: str = Field(..., description='name of file')
    text: str = Field(..., description='text to write to file')
    append: bool = Field(default=False, description='Whether to append to an existing file.')