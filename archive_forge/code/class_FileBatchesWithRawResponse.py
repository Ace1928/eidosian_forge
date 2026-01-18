from __future__ import annotations
import asyncio
from typing import List, Iterable
from typing_extensions import Literal
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import httpx
import sniffio
from .... import _legacy_response
from ....types import FileObject
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from ...._utils import (
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ....pagination import SyncCursorPage, AsyncCursorPage
from ...._base_client import (
from ....types.beta.vector_stores import (
class FileBatchesWithRawResponse:

    def __init__(self, file_batches: FileBatches) -> None:
        self._file_batches = file_batches
        self.create = _legacy_response.to_raw_response_wrapper(file_batches.create)
        self.retrieve = _legacy_response.to_raw_response_wrapper(file_batches.retrieve)
        self.cancel = _legacy_response.to_raw_response_wrapper(file_batches.cancel)
        self.list_files = _legacy_response.to_raw_response_wrapper(file_batches.list_files)