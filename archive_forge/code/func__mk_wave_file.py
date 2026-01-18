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
def _mk_wave_file(output_directory: Optional[str], sample_rate: float) -> Tuple[Optional[str], Optional[wave.Wave_write]]:
    """Create a new wave file and return the wave write object and filename."""
    if output_directory:
        with tempfile.NamedTemporaryFile(mode='bx', suffix='.wav', delete=False, dir=output_directory) as f:
            wav_file_name = f.name
        wav_file = wave.open(wav_file_name, 'wb')
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        return (wav_file_name, wav_file)
    return (None, None)