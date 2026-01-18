from __future__ import annotations
from .speech import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .translations import (
from .transcriptions import (
class AudioWithStreamingResponse:

    def __init__(self, audio: Audio) -> None:
        self._audio = audio

    @cached_property
    def transcriptions(self) -> TranscriptionsWithStreamingResponse:
        return TranscriptionsWithStreamingResponse(self._audio.transcriptions)

    @cached_property
    def translations(self) -> TranslationsWithStreamingResponse:
        return TranslationsWithStreamingResponse(self._audio.translations)

    @cached_property
    def speech(self) -> SpeechWithStreamingResponse:
        return SpeechWithStreamingResponse(self._audio.speech)