from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class ProtocolError(WebSocketException):
    """
    Raised when a frame breaks the protocol.

    """