import os
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.tools.file_management.utils import (
class DirectoryListingInput(BaseModel):
    """Input for ListDirectoryTool."""
    dir_path: str = Field(default='.', description='Subdirectory to list.')