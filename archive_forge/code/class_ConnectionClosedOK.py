from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class ConnectionClosedOK(ConnectionClosed):
    """
    Like :exc:`ConnectionClosed`, when the connection terminated properly.

    A close code with code 1000 (OK) or 1001 (going away) or without a code was
    received and sent.

    """