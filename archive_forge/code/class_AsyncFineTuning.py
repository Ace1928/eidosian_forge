from __future__ import annotations
from .jobs import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
class AsyncFineTuning(AsyncAPIResource):

    @cached_property
    def jobs(self) -> AsyncJobs:
        return AsyncJobs(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncFineTuningWithRawResponse:
        return AsyncFineTuningWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncFineTuningWithStreamingResponse:
        return AsyncFineTuningWithStreamingResponse(self)