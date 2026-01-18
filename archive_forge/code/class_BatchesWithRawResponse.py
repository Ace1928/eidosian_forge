from __future__ import annotations
from typing import Dict, Optional
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import Batch, batch_list_params, batch_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import (
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ..pagination import SyncCursorPage, AsyncCursorPage
from .._base_client import (
class BatchesWithRawResponse:

    def __init__(self, batches: Batches) -> None:
        self._batches = batches
        self.create = _legacy_response.to_raw_response_wrapper(batches.create)
        self.retrieve = _legacy_response.to_raw_response_wrapper(batches.retrieve)
        self.list = _legacy_response.to_raw_response_wrapper(batches.list)
        self.cancel = _legacy_response.to_raw_response_wrapper(batches.cancel)