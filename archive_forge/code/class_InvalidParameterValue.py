from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidParameterValue(NegotiationError):
    """
    Raised when a parameter value in an extension header is invalid.

    """

    def __init__(self, name: str, value: Optional[str]) -> None:
        self.name = name
        self.value = value

    def __str__(self) -> str:
        if self.value is None:
            return f'missing value for parameter {self.name}'
        elif self.value == '':
            return f'empty value for parameter {self.name}'
        else:
            return f'invalid value for parameter {self.name}: {self.value}'