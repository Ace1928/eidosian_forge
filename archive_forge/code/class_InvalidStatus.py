from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidStatus(InvalidHandshake):
    """
    Raised when a handshake response rejects the WebSocket upgrade.

    """

    def __init__(self, response: http11.Response) -> None:
        self.response = response

    def __str__(self) -> str:
        return f'server rejected WebSocket connection: HTTP {self.response.status_code:d}'