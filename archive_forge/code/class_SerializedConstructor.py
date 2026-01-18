from abc import ABC
from typing import (
from typing_extensions import NotRequired
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
class SerializedConstructor(BaseSerialized):
    """Serialized constructor."""
    type: Literal['constructor']
    kwargs: Dict[str, Any]