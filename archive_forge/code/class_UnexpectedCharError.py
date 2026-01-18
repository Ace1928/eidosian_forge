from __future__ import annotations
from typing import Collection
class UnexpectedCharError(ParseError):
    """
    An unexpected character was found during parsing.
    """

    def __init__(self, line: int, col: int, char: str) -> None:
        message = f'Unexpected character: {repr(char)}'
        super().__init__(line, col, message=message)