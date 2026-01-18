from __future__ import annotations
from .jobs import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
class AsyncFineTuningWithStreamingResponse:

    def __init__(self, fine_tuning: AsyncFineTuning) -> None:
        self._fine_tuning = fine_tuning

    @cached_property
    def jobs(self) -> AsyncJobsWithStreamingResponse:
        return AsyncJobsWithStreamingResponse(self._fine_tuning.jobs)