from __future__ import annotations
import argparse
import contextlib
import errno
import logging
import multiprocessing.pool
import operator
import signal
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from flake8 import defaults
from flake8 import exceptions
from flake8 import processor
from flake8 import utils
from flake8._compat import FSTRING_START
from flake8.discover_files import expand_paths
from flake8.options.parse_args import parse_args
from flake8.plugins.finder import Checkers
from flake8.plugins.finder import LoadedPlugin
from flake8.style_guide import StyleGuideManager
@staticmethod
def _extract_syntax_information(exception: Exception) -> tuple[int, int]:
    if len(exception.args) > 1 and exception.args[1] and (len(exception.args[1]) > 2):
        token = exception.args[1]
        row, column = token[1:3]
    elif isinstance(exception, tokenize.TokenError) and len(exception.args) == 2 and (len(exception.args[1]) == 2):
        token = ()
        row, column = exception.args[1]
    else:
        token = ()
        row, column = (1, 0)
    if column > 0 and token and isinstance(exception, SyntaxError) and (len(token) == 4):
        column_offset = 1
        row_offset = 0
        physical_line = token[3]
        if physical_line is not None:
            lines = physical_line.rstrip('\n').split('\n')
            row_offset = len(lines) - 1
            logical_line = lines[0]
            logical_line_length = len(logical_line)
            if column > logical_line_length:
                column = logical_line_length
        row -= row_offset
        column -= column_offset
    return (row, column)