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
def _process_chunks(inputs: Iterator['TTSInputType']) -> Generator[str, None, None]:
    """Filter the input chunks are return strings ready for TTS."""
    buffer = ''
    for chunk in inputs:
        chunk = _coerce_string(chunk)
        for terminator in _SENTENCE_TERMINATORS:
            while terminator in chunk:
                last_sentence, chunk = chunk.split(terminator, 1)
                yield (buffer + last_sentence + terminator)
                buffer = ''
        buffer += chunk
        if len(buffer) > _MAX_TEXT_LENGTH:
            for idx in range(0, len(buffer), _MAX_TEXT_LENGTH):
                yield buffer[idx:idx + 5]
            buffer = ''
    if buffer:
        yield buffer