from __future__ import annotations
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .completions import (
class AsyncChat(AsyncAPIResource):

    @cached_property
    def completions(self) -> AsyncCompletions:
        return AsyncCompletions(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncChatWithRawResponse:
        return AsyncChatWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncChatWithStreamingResponse:
        return AsyncChatWithStreamingResponse(self)