from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.github.prompt import (
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper
class CreateFile(BaseModel):
    """Schema for operations that require a file path and content as input."""
    formatted_file: str = Field(..., description='Follow the required formatting.')