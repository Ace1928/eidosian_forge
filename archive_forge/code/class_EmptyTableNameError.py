from __future__ import annotations
from typing import Collection
class EmptyTableNameError(ParseError):
    """
    An empty table name was found during parsing.
    """

    def __init__(self, line: int, col: int) -> None:
        message = 'Empty table name'
        super().__init__(line, col, message=message)