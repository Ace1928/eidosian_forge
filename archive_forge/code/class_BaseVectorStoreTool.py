import json
from typing import Any, Dict, Optional
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langchain_community.llms.openai import OpenAI
class BaseVectorStoreTool(BaseModel):
    """Base class for tools that use a VectorStore."""
    vectorstore: VectorStore = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config(BaseTool.Config):
        pass