from __future__ import annotations
import importlib
from typing import (
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import Literal
class IndexableBaseModel(BaseModel):
    """Allows a BaseModel to return its fields by string variable indexing."""

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)