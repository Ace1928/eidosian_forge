from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidHeader(InvalidHandshake):
    """
    Raised when an HTTP header doesn't have a valid format or value.

    """

    def __init__(self, name: str, value: Optional[str]=None) -> None:
        self.name = name
        self.value = value

    def __str__(self) -> str:
        if self.value is None:
            return f'missing {self.name} header'
        elif self.value == '':
            return f'empty {self.name} header'
        else:
            return f'invalid {self.name} header: {self.value}'