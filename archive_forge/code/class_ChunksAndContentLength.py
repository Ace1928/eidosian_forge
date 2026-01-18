from __future__ import annotations
import io
import typing
from base64 import b64encode
from enum import Enum
from ..exceptions import UnrewindableBodyError
from .util import to_bytes
class ChunksAndContentLength(typing.NamedTuple):
    chunks: typing.Iterable[bytes] | None
    content_length: int | None