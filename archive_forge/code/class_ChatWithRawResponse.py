from __future__ import annotations
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .completions import (
class ChatWithRawResponse:

    def __init__(self, chat: Chat) -> None:
        self._chat = chat

    @cached_property
    def completions(self) -> CompletionsWithRawResponse:
        return CompletionsWithRawResponse(self._chat.completions)