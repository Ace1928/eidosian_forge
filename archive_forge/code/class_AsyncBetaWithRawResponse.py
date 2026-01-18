from __future__ import annotations
from .threads import (
from ..._compat import cached_property
from .assistants import (
from ..._resource import SyncAPIResource, AsyncAPIResource
from .threads.threads import Threads, AsyncThreads
from .assistants.assistants import Assistants, AsyncAssistants
class AsyncBetaWithRawResponse:

    def __init__(self, beta: AsyncBeta) -> None:
        self._beta = beta

    @cached_property
    def assistants(self) -> AsyncAssistantsWithRawResponse:
        return AsyncAssistantsWithRawResponse(self._beta.assistants)

    @cached_property
    def threads(self) -> AsyncThreadsWithRawResponse:
        return AsyncThreadsWithRawResponse(self._beta.threads)