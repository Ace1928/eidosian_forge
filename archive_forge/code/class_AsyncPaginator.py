from __future__ import annotations
import json
import time
import uuid
import email
import asyncio
import inspect
import logging
import platform
import warnings
import email.utils
from types import TracebackType
from random import random
from typing import (
from functools import lru_cache
from typing_extensions import Literal, override, get_origin
import anyio
import httpx
import distro
import pydantic
from httpx import URL, Limits
from pydantic import PrivateAttr
from . import _exceptions
from ._qs import Querystring
from ._files import to_httpx_files, async_to_httpx_files
from ._types import (
from ._utils import is_dict, is_list, is_given, is_mapping
from ._compat import model_copy, model_dump
from ._models import GenericModel, FinalRequestOptions, validate_type, construct_type
from ._response import (
from ._constants import (
from ._streaming import Stream, SSEDecoder, AsyncStream, SSEBytesDecoder
from ._exceptions import (
from ._legacy_response import LegacyAPIResponse
class AsyncPaginator(Generic[_T, AsyncPageT]):

    def __init__(self, client: AsyncAPIClient, options: FinalRequestOptions, page_cls: Type[AsyncPageT], model: Type[_T]) -> None:
        self._model = model
        self._client = client
        self._options = options
        self._page_cls = page_cls

    def __await__(self) -> Generator[Any, None, AsyncPageT]:
        return self._get_page().__await__()

    async def _get_page(self) -> AsyncPageT:

        def _parser(resp: AsyncPageT) -> AsyncPageT:
            resp._set_private_attributes(model=self._model, options=self._options, client=self._client)
            return resp
        self._options.post_parser = _parser
        return await self._client.request(self._page_cls, self._options)

    async def __aiter__(self) -> AsyncIterator[_T]:
        page = cast(AsyncPageT, await self)
        async for item in page:
            yield item