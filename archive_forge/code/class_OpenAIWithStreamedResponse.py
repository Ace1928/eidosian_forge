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
class OpenAIWithStreamedResponse:

    def __init__(self, client: OpenAI) -> None:
        self.completions = resources.CompletionsWithStreamingResponse(client.completions)
        self.chat = resources.ChatWithStreamingResponse(client.chat)
        self.embeddings = resources.EmbeddingsWithStreamingResponse(client.embeddings)
        self.files = resources.FilesWithStreamingResponse(client.files)
        self.images = resources.ImagesWithStreamingResponse(client.images)
        self.audio = resources.AudioWithStreamingResponse(client.audio)
        self.moderations = resources.ModerationsWithStreamingResponse(client.moderations)
        self.models = resources.ModelsWithStreamingResponse(client.models)
        self.fine_tuning = resources.FineTuningWithStreamingResponse(client.fine_tuning)
        self.beta = resources.BetaWithStreamingResponse(client.beta)