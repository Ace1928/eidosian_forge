from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class RedirectHandshake(InvalidHandshake):
    """
    Raised when a handshake gets redirected.

    This exception is an implementation detail.

    """

    def __init__(self, uri: str) -> None:
        self.uri = uri

    def __str__(self) -> str:
        return f'redirect to {self.uri}'