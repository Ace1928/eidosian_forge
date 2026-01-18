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
def _consumer() -> None:
    """Consume the input with transform."""
    for val in self.transform(_input_iterator()):
        out_queue.put_nowait(val)
    out_queue.put_nowait(_TRANSFORM_END)