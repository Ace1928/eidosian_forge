from __future__ import annotations
import importlib
from typing import (
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import Literal
def _convert_message_chunk_to_delta(chunk: BaseMessageChunk, i: int) -> Dict[str, Any]:
    _dict = _convert_message_chunk(chunk, i)
    return {'choices': [{'delta': _dict}]}