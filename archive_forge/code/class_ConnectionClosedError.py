from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class ConnectionClosedError(ConnectionClosed):
    """
    Like :exc:`ConnectionClosed`, when the connection terminated with an error.

    A close frame with a code other than 1000 (OK) or 1001 (going away) was
    received or sent, or the closing handshake didn't complete properly.

    """