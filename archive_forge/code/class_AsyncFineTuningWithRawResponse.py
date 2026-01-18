from __future__ import annotations
from .jobs import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
class AsyncFineTuningWithRawResponse:

    def __init__(self, fine_tuning: AsyncFineTuning) -> None:
        self._fine_tuning = fine_tuning

    @cached_property
    def jobs(self) -> AsyncJobsWithRawResponse:
        return AsyncJobsWithRawResponse(self._fine_tuning.jobs)