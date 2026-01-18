from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidURI(WebSocketException):
    """
    Raised when connecting to a URI that isn't a valid WebSocket URI.

    """

    def __init__(self, uri: str, msg: str) -> None:
        self.uri = uri
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.uri} isn't a valid URI: {self.msg}"