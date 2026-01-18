from abc import ABC
from typing import (
from typing_extensions import NotRequired
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
class BaseSerialized(TypedDict):
    """Base class for serialized objects."""
    lc: int
    id: List[str]
    name: NotRequired[str]
    graph: NotRequired[Dict[str, Any]]