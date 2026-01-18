from __future__ import annotations
import os
import inspect
import logging
import datetime
import functools
from types import TracebackType
from typing import (
from typing_extensions import Awaitable, ParamSpec, override, get_origin
import anyio
import httpx
import pydantic
from ._types import NoneType
from ._utils import is_given, extract_type_arg, is_annotated_type, extract_type_var_from_base
from ._models import BaseModel, is_basemodel
from ._constants import RAW_RESPONSE_HEADER, OVERRIDE_CAST_TO_HEADER
from ._streaming import Stream, AsyncStream, is_stream_class_type, extract_stream_chunk_type
from ._exceptions import OpenAIError, APIResponseValidationError
class ResponseContextManager(Generic[_APIResponseT]):
    """Context manager for ensuring that a request is not made
    until it is entered and that the response will always be closed
    when the context manager exits
    """

    def __init__(self, request_func: Callable[[], _APIResponseT]) -> None:
        self._request_func = request_func
        self.__response: _APIResponseT | None = None

    def __enter__(self) -> _APIResponseT:
        self.__response = self._request_func()
        return self.__response

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, exc_tb: TracebackType | None) -> None:
        if self.__response is not None:
            self.__response.close()