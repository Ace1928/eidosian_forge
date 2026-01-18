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
class RivaCommonConfigMixin(BaseModel):
    """A collection of common Riva settings."""
    encoding: RivaAudioEncoding = Field(default=RivaAudioEncoding.LINEAR_PCM, description='The encoding on the audio stream.')
    sample_rate_hertz: int = Field(default=8000, description='The sample rate frequency of audio stream.')
    language_code: str = Field(default='en-US', description='The [BCP-47 language code](https://www.rfc-editor.org/rfc/bcp/bcp47.txt) for the target language.')