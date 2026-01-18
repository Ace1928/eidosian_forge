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
def _import_riva_client() -> 'riva.client':
    """Import the riva client and raise an error on failure."""
    try:
        import riva.client
    except ImportError as err:
        raise ImportError('Could not import the NVIDIA Riva client library. Please install it with `pip install nvidia-riva-client`.') from err
    return riva.client