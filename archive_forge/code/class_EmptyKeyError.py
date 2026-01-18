from __future__ import annotations
from typing import Collection
class EmptyKeyError(ParseError):
    """
    An empty key was found during parsing.
    """

    def __init__(self, line: int, col: int) -> None:
        message = 'Empty key'
        super().__init__(line, col, message=message)