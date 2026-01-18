from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidStatusCode(InvalidHandshake):
    """
    Raised when a handshake response status code is invalid.

    """

    def __init__(self, status_code: int, headers: datastructures.Headers) -> None:
        self.status_code = status_code
        self.headers = headers

    def __str__(self) -> str:
        return f'server rejected WebSocket connection: HTTP {self.status_code}'