from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
class ByteParser:
    """Parse bytecode to understand the structure of code."""

    def __init__(self, text: str, code: CodeType | None=None, filename: str | None=None) -> None:
        self.text = text
        if code is not None:
            self.code = code
        else:
            assert filename is not None
            try:
                self.code = compile(text, filename, 'exec', dont_inherit=True)
            except SyntaxError as synerr:
                raise NotPython("Couldn't parse '%s' as Python source: '%s' at line %d" % (filename, synerr.msg, synerr.lineno or 0)) from synerr

    def child_parsers(self) -> Iterable[ByteParser]:
        """Iterate over all the code objects nested within this one.

        The iteration includes `self` as its first value.

        """
        return (ByteParser(self.text, code=c) for c in code_objects(self.code))

    def _line_numbers(self) -> Iterable[TLineNo]:
        """Yield the line numbers possible in this code object.

        Uses co_lnotab described in Python/compile.c to find the
        line numbers.  Produces a sequence: l0, l1, ...
        """
        if hasattr(self.code, 'co_lines'):
            for _, _, line in self.code.co_lines():
                if line:
                    yield line
        else:
            byte_increments = self.code.co_lnotab[0::2]
            line_increments = self.code.co_lnotab[1::2]
            last_line_num = None
            line_num = self.code.co_firstlineno
            byte_num = 0
            for byte_incr, line_incr in zip(byte_increments, line_increments):
                if byte_incr:
                    if line_num != last_line_num:
                        yield line_num
                        last_line_num = line_num
                    byte_num += byte_incr
                if line_incr >= 128:
                    line_incr -= 256
                line_num += line_incr
            if line_num != last_line_num:
                yield line_num

    def _find_statements(self) -> Iterable[TLineNo]:
        """Find the statements in `self.code`.

        Produce a sequence of line numbers that start statements.  Recurses
        into all code objects reachable from `self.code`.

        """
        for bp in self.child_parsers():
            yield from bp._line_numbers()