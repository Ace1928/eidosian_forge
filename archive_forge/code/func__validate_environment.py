import asyncio
import logging
import pathlib
import queue
import tempfile
import threading
import wave
from enum import Enum
from typing import (
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import (
from langchain_core.runnables import RunnableConfig, RunnableSerializable
@root_validator(pre=True)
@classmethod
def _validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the Python environment and input arguments."""
    _ = _import_riva_client()
    return values