from __future__ import annotations
from typing import List, Optional
from typing_extensions import Literal
import httpx
from .... import _legacy_response
from .files import (
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ...._utils import (
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .file_batches import (
from ....pagination import SyncCursorPage, AsyncCursorPage
from ....types.beta import (
from ...._base_client import (
class VectorStoresWithStreamingResponse:

    def __init__(self, vector_stores: VectorStores) -> None:
        self._vector_stores = vector_stores
        self.create = to_streamed_response_wrapper(vector_stores.create)
        self.retrieve = to_streamed_response_wrapper(vector_stores.retrieve)
        self.update = to_streamed_response_wrapper(vector_stores.update)
        self.list = to_streamed_response_wrapper(vector_stores.list)
        self.delete = to_streamed_response_wrapper(vector_stores.delete)

    @cached_property
    def files(self) -> FilesWithStreamingResponse:
        return FilesWithStreamingResponse(self._vector_stores.files)

    @cached_property
    def file_batches(self) -> FileBatchesWithStreamingResponse:
        return FileBatchesWithStreamingResponse(self._vector_stores.file_batches)