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