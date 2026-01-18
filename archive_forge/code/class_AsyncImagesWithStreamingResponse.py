from __future__ import annotations
from typing import Union, Mapping, Optional, cast
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import (
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from .._utils import (
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .._base_client import (
class AsyncImagesWithStreamingResponse:

    def __init__(self, images: AsyncImages) -> None:
        self._images = images
        self.create_variation = async_to_streamed_response_wrapper(images.create_variation)
        self.edit = async_to_streamed_response_wrapper(images.edit)
        self.generate = async_to_streamed_response_wrapper(images.generate)