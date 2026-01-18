from __future__ import annotations
from typing import Any, Dict, List, Literal
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs.generation import Generation
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts
Get the namespace of the langchain object.