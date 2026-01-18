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
class RivaASR(RivaAuthMixin, RivaCommonConfigMixin, RunnableSerializable[ASRInputType, ASROutputType]):
    """A runnable that performs Automatic Speech Recognition (ASR) using NVIDIA Riva."""
    name: str = 'nvidia_riva_asr'
    description: str = 'A Runnable for converting audio bytes to a string.This is useful for feeding an audio stream into a chain andpreprocessing that audio to create an LLM prompt.'
    audio_channel_count: int = Field(1, description='The number of audio channels in the input audio stream.')
    profanity_filter: bool = Field(True, description='Controls whether or not Riva should attempt to filter profanity out of the transcribed text.')
    enable_automatic_punctuation: bool = Field(True, description='Controls whether Riva should attempt to correct senetence puncuation in the transcribed text.')

    @root_validator(pre=True)
    @classmethod
    def _validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the Python environment and input arguments."""
        _ = _import_riva_client()
        return values

    @property
    def config(self) -> 'riva.client.StreamingRecognitionConfig':
        """Create and return the riva config object."""
        riva_client = _import_riva_client()
        return riva_client.StreamingRecognitionConfig(interim_results=True, config=riva_client.RecognitionConfig(encoding=self.encoding, sample_rate_hertz=self.sample_rate_hertz, audio_channel_count=self.audio_channel_count, max_alternatives=1, profanity_filter=self.profanity_filter, enable_automatic_punctuation=self.enable_automatic_punctuation, language_code=self.language_code))

    def _get_service(self) -> 'riva.client.ASRService':
        """Connect to the riva service and return the a client object."""
        riva_client = _import_riva_client()
        try:
            return riva_client.ASRService(self.auth)
        except Exception as err:
            raise ValueError('Error raised while connecting to the Riva ASR server.') from err

    def invoke(self, input: ASRInputType, _: Optional[RunnableConfig]=None) -> ASROutputType:
        """Transcribe the audio bytes into a string with Riva."""
        if not input.running:
            service = self._get_service()
            responses = service.streaming_response_generator(audio_chunks=input, streaming_config=self.config)
            input.register(responses)
        full_response: List[str] = []
        while not input.complete:
            with input.output.not_empty:
                ready = input.output.not_empty.wait(0.1)
            if ready:
                while not input.output.empty():
                    try:
                        full_response += [input.output.get_nowait()]
                    except queue.Empty:
                        continue
                    input.output.task_done()
                _LOGGER.debug('Riva ASR returning: %s', repr(full_response))
                return ' '.join(full_response).strip()
        return ''