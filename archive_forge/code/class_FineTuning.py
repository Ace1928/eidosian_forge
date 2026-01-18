from __future__ import annotations
from .jobs import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
class FineTuning(SyncAPIResource):

    @cached_property
    def jobs(self) -> Jobs:
        return Jobs(self._client)

    @cached_property
    def with_raw_response(self) -> FineTuningWithRawResponse:
        return FineTuningWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> FineTuningWithStreamingResponse:
        return FineTuningWithStreamingResponse(self)