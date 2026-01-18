import fnmatch
import os
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.tools.file_management.utils import (
class FileSearchInput(BaseModel):
    """Input for FileSearchTool."""
    dir_path: str = Field(default='.', description='Subdirectory to search in.')
    pattern: str = Field(..., description='Unix shell regex, where * matches everything.')