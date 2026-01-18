from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.github.prompt import (
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper
class DirectoryPath(BaseModel):
    """Schema for operations that require a directory path as input."""
    input: str = Field('', description='The path of the directory, e.g. `some_dir/inner_dir`. Only input a string, do not include the parameter name.')