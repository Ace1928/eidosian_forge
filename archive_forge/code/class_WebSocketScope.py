from __future__ import annotations
import sys
import types
from typing import (
class WebSocketScope(TypedDict):
    type: Literal['websocket']
    asgi: ASGIVersions
    http_version: str
    scheme: str
    path: str
    raw_path: bytes
    query_string: bytes
    root_path: str
    headers: Iterable[tuple[bytes, bytes]]
    client: tuple[str, int] | None
    server: tuple[str, int | None] | None
    subprotocols: Iterable[str]
    state: NotRequired[dict[str, Any]]
    extensions: NotRequired[dict[str, dict[object, object]]]