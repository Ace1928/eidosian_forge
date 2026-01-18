from __future__ import annotations
from .speech import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .translations import (
from .transcriptions import (
class Audio(SyncAPIResource):

    @cached_property
    def transcriptions(self) -> Transcriptions:
        return Transcriptions(self._client)

    @cached_property
    def translations(self) -> Translations:
        return Translations(self._client)

    @cached_property
    def speech(self) -> Speech:
        return Speech(self._client)

    @cached_property
    def with_raw_response(self) -> AudioWithRawResponse:
        return AudioWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AudioWithStreamingResponse:
        return AudioWithStreamingResponse(self)