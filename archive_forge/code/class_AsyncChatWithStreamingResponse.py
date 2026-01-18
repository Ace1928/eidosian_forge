from __future__ import annotations
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .completions import (
class AsyncChatWithStreamingResponse:

    def __init__(self, chat: AsyncChat) -> None:
        self._chat = chat

    @cached_property
    def completions(self) -> AsyncCompletionsWithStreamingResponse:
        return AsyncCompletionsWithStreamingResponse(self._chat.completions)