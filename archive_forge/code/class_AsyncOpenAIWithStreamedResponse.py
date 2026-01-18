from __future__ import annotations
import os
from typing import Any, Union, Mapping
from typing_extensions import Self, override
import httpx
from . import resources, _exceptions
from ._qs import Querystring
from ._types import (
from ._utils import (
from ._version import __version__
from ._streaming import Stream as Stream, AsyncStream as AsyncStream
from ._exceptions import OpenAIError, APIStatusError
from ._base_client import (
class AsyncOpenAIWithStreamedResponse:

    def __init__(self, client: AsyncOpenAI) -> None:
        self.completions = resources.AsyncCompletionsWithStreamingResponse(client.completions)
        self.chat = resources.AsyncChatWithStreamingResponse(client.chat)
        self.embeddings = resources.AsyncEmbeddingsWithStreamingResponse(client.embeddings)
        self.files = resources.AsyncFilesWithStreamingResponse(client.files)
        self.images = resources.AsyncImagesWithStreamingResponse(client.images)
        self.audio = resources.AsyncAudioWithStreamingResponse(client.audio)
        self.moderations = resources.AsyncModerationsWithStreamingResponse(client.moderations)
        self.models = resources.AsyncModelsWithStreamingResponse(client.models)
        self.fine_tuning = resources.AsyncFineTuningWithStreamingResponse(client.fine_tuning)
        self.beta = resources.AsyncBetaWithStreamingResponse(client.beta)