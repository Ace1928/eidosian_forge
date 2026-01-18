from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidHandshake(WebSocketException):
    """
    Raised during the handshake when the WebSocket connection fails.

    """