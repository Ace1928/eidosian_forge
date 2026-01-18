from __future__ import annotations
from typing import Collection
class InvalidControlChar(ParseError):

    def __init__(self, line: int, col: int, char: int, type: str) -> None:
        display_code = '\\u00'
        if char < 16:
            display_code += '0'
        display_code += hex(char)[2:]
        message = f'Control characters (codes less than 0x1f and 0x7f) are not allowed in {type}, use {display_code} instead'
        super().__init__(line, col, message=message)