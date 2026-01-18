from __future__ import annotations
import os as _os
from typing_extensions import override
from . import types
from ._types import NOT_GIVEN, NoneType, NotGiven, Transport, ProxiesTypes
from ._utils import file_from_path
from ._client import Client, OpenAI, Stream, Timeout, Transport, AsyncClient, AsyncOpenAI, AsyncStream, RequestOptions
from ._models import BaseModel
from ._version import __title__, __version__
from ._response import APIResponse as APIResponse, AsyncAPIResponse as AsyncAPIResponse
from ._exceptions import (
from ._utils._logs import setup_logging as _setup_logging
from .lib import azure as _azure
from .version import VERSION as VERSION
from .lib.azure import AzureOpenAI as AzureOpenAI, AsyncAzureOpenAI as AsyncAzureOpenAI
from .lib._old_api import *
from .lib.streaming import (
import typing as _t
import typing_extensions as _te
import httpx as _httpx
from ._base_client import DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES
from ._module_client import (
class _ModuleClient(OpenAI):

    @property
    @override
    def api_key(self) -> str | None:
        return api_key

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        global api_key
        api_key = value

    @property
    @override
    def organization(self) -> str | None:
        return organization

    @organization.setter
    def organization(self, value: str | None) -> None:
        global organization
        organization = value

    @property
    @override
    def base_url(self) -> _httpx.URL:
        if base_url is not None:
            return _httpx.URL(base_url)
        return super().base_url

    @base_url.setter
    def base_url(self, url: _httpx.URL | str) -> None:
        super().base_url = url

    @property
    @override
    def timeout(self) -> float | Timeout | None:
        return timeout

    @timeout.setter
    def timeout(self, value: float | Timeout | None) -> None:
        global timeout
        timeout = value

    @property
    @override
    def max_retries(self) -> int:
        return max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        global max_retries
        max_retries = value

    @property
    @override
    def _custom_headers(self) -> _t.Mapping[str, str] | None:
        return default_headers

    @_custom_headers.setter
    def _custom_headers(self, value: _t.Mapping[str, str] | None) -> None:
        global default_headers
        default_headers = value

    @property
    @override
    def _custom_query(self) -> _t.Mapping[str, object] | None:
        return default_query

    @_custom_query.setter
    def _custom_query(self, value: _t.Mapping[str, object] | None) -> None:
        global default_query
        default_query = value

    @property
    @override
    def _client(self) -> _httpx.Client:
        return http_client or super()._client

    @_client.setter
    def _client(self, value: _httpx.Client) -> None:
        global http_client
        http_client = value