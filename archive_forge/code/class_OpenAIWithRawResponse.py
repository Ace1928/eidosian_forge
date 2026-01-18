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
class OpenAIWithRawResponse:

    def __init__(self, client: OpenAI) -> None:
        self.completions = resources.CompletionsWithRawResponse(client.completions)
        self.chat = resources.ChatWithRawResponse(client.chat)
        self.embeddings = resources.EmbeddingsWithRawResponse(client.embeddings)
        self.files = resources.FilesWithRawResponse(client.files)
        self.images = resources.ImagesWithRawResponse(client.images)
        self.audio = resources.AudioWithRawResponse(client.audio)
        self.moderations = resources.ModerationsWithRawResponse(client.moderations)
        self.models = resources.ModelsWithRawResponse(client.models)
        self.fine_tuning = resources.FineTuningWithRawResponse(client.fine_tuning)
        self.beta = resources.BetaWithRawResponse(client.beta)