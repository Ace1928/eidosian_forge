from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidHeaderValue(InvalidHeader):
    """
    Raised when an HTTP header has a wrong value.

    The format of the header is correct but a value isn't acceptable.

    """