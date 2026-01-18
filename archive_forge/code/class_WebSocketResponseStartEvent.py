from __future__ import annotations
import sys
import types
from typing import (
class WebSocketResponseStartEvent(TypedDict):
    type: Literal['websocket.http.response.start']
    status: int
    headers: Iterable[tuple[bytes, bytes]]