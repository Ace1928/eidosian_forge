from typing import List
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.llms.openai import OpenAI
from langchain_community.tools.vectorstore.tool import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.vectorstores import VectorStore
from langchain.tools import BaseTool
class VectorStoreInfo(BaseModel):
    """Information about a VectorStore."""
    vectorstore: VectorStore = Field(exclude=True)
    name: str
    description: str

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True