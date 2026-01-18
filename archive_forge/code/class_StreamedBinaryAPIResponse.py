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
class StreamedBinaryAPIResponse(APIResponse[bytes]):

    def stream_to_file(self, file: str | os.PathLike[str], *, chunk_size: int | None=None) -> None:
        """Streams the output to the given file.

        Accepts a filename or any path-like object, e.g. pathlib.Path
        """
        with open(file, mode='wb') as f:
            for data in self.iter_bytes(chunk_size):
                f.write(data)