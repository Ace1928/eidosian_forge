from __future__ import annotations
import sys
import types
from typing import (
class HTTPResponseStartEvent(TypedDict):
    type: Literal['http.response.start']
    status: int
    headers: NotRequired[Iterable[tuple[bytes, bytes]]]
    trailers: NotRequired[bool]