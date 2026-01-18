from __future__ import annotations
from .jobs import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
class FineTuningWithRawResponse:

    def __init__(self, fine_tuning: FineTuning) -> None:
        self._fine_tuning = fine_tuning

    @cached_property
    def jobs(self) -> JobsWithRawResponse:
        return JobsWithRawResponse(self._fine_tuning.jobs)