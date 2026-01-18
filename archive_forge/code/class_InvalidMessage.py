from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidMessage(InvalidHandshake):
    """
    Raised when a handshake request or response is malformed.

    """