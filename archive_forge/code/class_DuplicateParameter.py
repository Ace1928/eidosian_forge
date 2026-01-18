from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class DuplicateParameter(NegotiationError):
    """
    Raised when a parameter name is repeated in an extension header.

    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f'duplicate parameter: {self.name}'