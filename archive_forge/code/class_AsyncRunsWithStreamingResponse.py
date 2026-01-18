from __future__ import annotations
from typing import Iterable, Optional, overload
from functools import partial
from typing_extensions import Literal
import httpx
from ..... import _legacy_response
from .steps import (
from ....._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ....._utils import (
from ....._compat import cached_property
from ....._resource import SyncAPIResource, AsyncAPIResource
from ....._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ....._streaming import Stream, AsyncStream
from .....pagination import SyncCursorPage, AsyncCursorPage
from .....types.beta import AssistantToolParam, AssistantStreamEvent
from ....._base_client import (
from .....lib.streaming import (
from .....types.beta.threads import (
class AsyncRunsWithStreamingResponse:

    def __init__(self, runs: AsyncRuns) -> None:
        self._runs = runs
        self.create = async_to_streamed_response_wrapper(runs.create)
        self.retrieve = async_to_streamed_response_wrapper(runs.retrieve)
        self.update = async_to_streamed_response_wrapper(runs.update)
        self.list = async_to_streamed_response_wrapper(runs.list)
        self.cancel = async_to_streamed_response_wrapper(runs.cancel)
        self.submit_tool_outputs = async_to_streamed_response_wrapper(runs.submit_tool_outputs)

    @cached_property
    def steps(self) -> AsyncStepsWithStreamingResponse:
        return AsyncStepsWithStreamingResponse(self._runs.steps)