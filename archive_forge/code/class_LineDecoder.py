from __future__ import annotations
import codecs
import io
import typing
import zlib
from ._compat import brotli
from ._exceptions import DecodingError
class LineDecoder:
    """
    Handles incrementally reading lines from text.

    Has the same behaviour as the stdllib splitlines,
    but handling the input iteratively.
    """

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.trailing_cr: bool = False

    def decode(self, text: str) -> list[str]:
        NEWLINE_CHARS = '\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029'
        if self.trailing_cr:
            text = '\r' + text
            self.trailing_cr = False
        if text.endswith('\r'):
            self.trailing_cr = True
            text = text[:-1]
        if not text:
            return []
        trailing_newline = text[-1] in NEWLINE_CHARS
        lines = text.splitlines()
        if len(lines) == 1 and (not trailing_newline):
            self.buffer.append(lines[0])
            return []
        if self.buffer:
            lines = [''.join(self.buffer) + lines[0]] + lines[1:]
            self.buffer = []
        if not trailing_newline:
            self.buffer = [lines.pop()]
        return lines

    def flush(self) -> list[str]:
        if not self.buffer and (not self.trailing_cr):
            return []
        lines = [''.join(self.buffer)]
        self.buffer = []
        self.trailing_cr = False
        return lines