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
class BinaryAPIResponse(APIResponse[bytes]):
    """Subclass of APIResponse providing helpers for dealing with binary data.

    Note: If you want to stream the response data instead of eagerly reading it
    all at once then you should use `.with_streaming_response` when making
    the API request, e.g. `.with_streaming_response.get_binary_response()`
    """

    def write_to_file(self, file: str | os.PathLike[str]) -> None:
        """Write the output to the given file.

        Accepts a filename or any path-like object, e.g. pathlib.Path

        Note: if you want to stream the data to the file instead of writing
        all at once then you should use `.with_streaming_response` when making
        the API request, e.g. `.with_streaming_response.get_binary_response()`
        """
        with open(file, mode='wb') as f:
            for data in self.iter_bytes():
                f.write(data)