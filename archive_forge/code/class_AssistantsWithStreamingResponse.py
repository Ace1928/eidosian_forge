from __future__ import annotations
from typing import List, Iterable, Optional
from typing_extensions import Literal
import httpx
from .... import _legacy_response
from .files import (
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ...._utils import (
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ....pagination import SyncCursorPage, AsyncCursorPage
from ....types.beta import (
from ...._base_client import (
class AssistantsWithStreamingResponse:

    def __init__(self, assistants: Assistants) -> None:
        self._assistants = assistants
        self.create = to_streamed_response_wrapper(assistants.create)
        self.retrieve = to_streamed_response_wrapper(assistants.retrieve)
        self.update = to_streamed_response_wrapper(assistants.update)
        self.list = to_streamed_response_wrapper(assistants.list)
        self.delete = to_streamed_response_wrapper(assistants.delete)

    @cached_property
    def files(self) -> FilesWithStreamingResponse:
        return FilesWithStreamingResponse(self._assistants.files)