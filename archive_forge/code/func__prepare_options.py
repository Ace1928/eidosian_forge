from __future__ import annotations
import os
import inspect
from typing import Any, Union, Mapping, TypeVar, Callable, Awaitable, overload
from typing_extensions import Self, override
import httpx
from .._types import NOT_GIVEN, Omit, Timeout, NotGiven
from .._utils import is_given, is_mapping
from .._client import OpenAI, AsyncOpenAI
from .._models import FinalRequestOptions
from .._streaming import Stream, AsyncStream
from .._exceptions import OpenAIError
from .._base_client import DEFAULT_MAX_RETRIES, BaseClient
@override
def _prepare_options(self, options: FinalRequestOptions) -> None:
    headers: dict[str, str | Omit] = {**options.headers} if is_given(options.headers) else {}
    options.headers = headers
    azure_ad_token = self._get_azure_ad_token()
    if azure_ad_token is not None:
        if headers.get('Authorization') is None:
            headers['Authorization'] = f'Bearer {azure_ad_token}'
    elif self.api_key is not API_KEY_SENTINEL:
        if headers.get('api-key') is None:
            headers['api-key'] = self.api_key
    else:
        raise ValueError('Unable to handle auth')
    return super()._prepare_options(options)