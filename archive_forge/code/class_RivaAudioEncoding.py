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
class RivaAudioEncoding(str, Enum):
    """An enum of the possible choices for Riva audio encoding.

    The list of types exposed by the Riva GRPC Protobuf files can be found
    with the following commands:
    ```python
    import riva.client
    print(riva.client.AudioEncoding.keys())  # noqa: T201
    ```
    """
    ALAW = 'ALAW'
    ENCODING_UNSPECIFIED = 'ENCODING_UNSPECIFIED'
    FLAC = 'FLAC'
    LINEAR_PCM = 'LINEAR_PCM'
    MULAW = 'MULAW'
    OGGOPUS = 'OGGOPUS'

    @classmethod
    def from_wave_format_code(cls, format_code: int) -> 'RivaAudioEncoding':
        """Return the audio encoding specified by the format code in the wave file.

        ref: https://mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
        """
        try:
            return {1: cls.LINEAR_PCM, 6: cls.ALAW, 7: cls.MULAW}[format_code]
        except KeyError as err:
            raise NotImplementedError(f'The following wave file format code is not supported by Riva: {format_code}') from err

    @property
    def riva_pb2(self) -> 'riva.client.AudioEncoding':
        """Returns the Riva API object for the encoding."""
        riva_client = _import_riva_client()
        return getattr(riva_client.AudioEncoding, self)