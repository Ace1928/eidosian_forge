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
class MutuallyExclusiveAuthError(OpenAIError):

    def __init__(self) -> None:
        super().__init__('The `api_key`, `azure_ad_token` and `azure_ad_token_provider` arguments are mutually exclusive; Only one can be passed at a time')