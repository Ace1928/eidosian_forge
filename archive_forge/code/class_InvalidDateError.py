from __future__ import annotations
from typing import Collection
class InvalidDateError(ParseError):
    """
    A date field was improperly specified.
    """

    def __init__(self, line: int, col: int) -> None:
        message = 'Invalid date'
        super().__init__(line, col, message=message)