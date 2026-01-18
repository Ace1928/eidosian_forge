from __future__ import annotations
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .completions import (
class AsyncChatWithRawResponse:

    def __init__(self, chat: AsyncChat) -> None:
        self._chat = chat

    @cached_property
    def completions(self) -> AsyncCompletionsWithRawResponse:
        return AsyncCompletionsWithRawResponse(self._chat.completions)