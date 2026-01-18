from __future__ import annotations
import typing as t
from types import TracebackType
from urllib.parse import urlparse
from warnings import warn
from ..datastructures import Headers
from ..http import is_entity_header
from ..wsgi import FileWrapper
class GuardedWrite:

    def __init__(self, write: t.Callable[[bytes], object], chunks: list[int]) -> None:
        self._write = write
        self._chunks = chunks

    def __call__(self, s: bytes) -> None:
        check_type('write()', s, bytes)
        self._write(s)
        self._chunks.append(len(s))