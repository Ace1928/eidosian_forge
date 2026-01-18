from __future__ import annotations
from .threads import (
from ..._compat import cached_property
from .assistants import (
from ..._resource import SyncAPIResource, AsyncAPIResource
from .threads.threads import Threads, AsyncThreads
from .assistants.assistants import Assistants, AsyncAssistants
class AsyncBetaWithStreamingResponse:

    def __init__(self, beta: AsyncBeta) -> None:
        self._beta = beta

    @cached_property
    def assistants(self) -> AsyncAssistantsWithStreamingResponse:
        return AsyncAssistantsWithStreamingResponse(self._beta.assistants)

    @cached_property
    def threads(self) -> AsyncThreadsWithStreamingResponse:
        return AsyncThreadsWithStreamingResponse(self._beta.threads)