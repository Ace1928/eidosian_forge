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
class AsyncOpenAIWithRawResponse:

    def __init__(self, client: AsyncOpenAI) -> None:
        self.completions = resources.AsyncCompletionsWithRawResponse(client.completions)
        self.chat = resources.AsyncChatWithRawResponse(client.chat)
        self.embeddings = resources.AsyncEmbeddingsWithRawResponse(client.embeddings)
        self.files = resources.AsyncFilesWithRawResponse(client.files)
        self.images = resources.AsyncImagesWithRawResponse(client.images)
        self.audio = resources.AsyncAudioWithRawResponse(client.audio)
        self.moderations = resources.AsyncModerationsWithRawResponse(client.moderations)
        self.models = resources.AsyncModelsWithRawResponse(client.models)
        self.fine_tuning = resources.AsyncFineTuningWithRawResponse(client.fine_tuning)
        self.beta = resources.AsyncBetaWithRawResponse(client.beta)