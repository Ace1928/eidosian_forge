from __future__ import annotations
import io
import itertools
import sys
import typing
from .._models import Request, Response
from .._types import SyncByteStream
from .base import BaseTransport
def _skip_leading_empty_chunks(body: typing.Iterable[_T]) -> typing.Iterable[_T]:
    body = iter(body)
    for chunk in body:
        if chunk:
            return itertools.chain([chunk], body)
    return []