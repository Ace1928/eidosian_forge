from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidOrigin(InvalidHeader):
    """
    Raised when the Origin header in a request isn't allowed.

    """

    def __init__(self, origin: Optional[str]) -> None:
        super().__init__('Origin', origin)