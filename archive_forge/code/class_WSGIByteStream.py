from __future__ import annotations
import io
import itertools
import sys
import typing
from .._models import Request, Response
from .._types import SyncByteStream
from .base import BaseTransport
class WSGIByteStream(SyncByteStream):

    def __init__(self, result: typing.Iterable[bytes]) -> None:
        self._close = getattr(result, 'close', None)
        self._result = _skip_leading_empty_chunks(result)

    def __iter__(self) -> typing.Iterator[bytes]:
        for part in self._result:
            yield part

    def close(self) -> None:
        if self._close is not None:
            self._close()