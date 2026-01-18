from __future__ import annotations
from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import contextlib
from dataclasses import dataclass, field
import functools
from .compat import io
import itertools
import os
import re
import sys
from typing import Iterable
class MultilineContinuationError(ParsingError):
    """Raised when a key without value is followed by continuation line"""

    def __init__(self, filename, lineno, line):
        Error.__init__(self, 'Key without value continued with an indented line.\nfile: %r, line: %d\n%r' % (filename, lineno, line))
        self.source = filename
        self.lineno = lineno
        self.line = line
        self.args = (filename, lineno, line)