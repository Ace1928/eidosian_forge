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
def _load_client() -> OpenAI:
    global _client
    if _client is None:
        global api_type, azure_endpoint, azure_ad_token, api_version
        if azure_endpoint is None:
            azure_endpoint = _os.environ.get('AZURE_OPENAI_ENDPOINT')
        if azure_ad_token is None:
            azure_ad_token = _os.environ.get('AZURE_OPENAI_AD_TOKEN')
        if api_version is None:
            api_version = _os.environ.get('OPENAI_API_VERSION')
        if api_type is None:
            has_openai = _has_openai_credentials()
            has_azure = _has_azure_credentials()
            has_azure_ad = _has_azure_ad_credentials()
            if has_openai and (has_azure or has_azure_ad):
                raise _AmbiguousModuleClientUsageError()
            if (azure_ad_token is not None or azure_ad_token_provider is not None) and _os.environ.get('AZURE_OPENAI_API_KEY') is not None:
                raise _AmbiguousModuleClientUsageError()
            if has_azure or has_azure_ad:
                api_type = 'azure'
            else:
                api_type = 'openai'
        if api_type == 'azure':
            _client = _AzureModuleClient(api_version=api_version, azure_endpoint=azure_endpoint, api_key=api_key, azure_ad_token=azure_ad_token, azure_ad_token_provider=azure_ad_token_provider, organization=organization, base_url=base_url, timeout=timeout, max_retries=max_retries, default_headers=default_headers, default_query=default_query, http_client=http_client)
            return _client
        _client = _ModuleClient(api_key=api_key, organization=organization, base_url=base_url, timeout=timeout, max_retries=max_retries, default_headers=default_headers, default_query=default_query, http_client=http_client)
        return _client
    return _client