import io
import logging
import os
import pkgutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import Logger
from typing import IO, Any, Iterable, Iterator, List, Optional, Tuple, Union, cast
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.tokenize import GoodTokenInfo
from blib2to3.pytree import NL
from . import grammar, parse, pgen, token, tokenize
def _partially_consume_prefix(self, prefix: str, column: int) -> Tuple[str, str]:
    lines: List[str] = []
    current_line = ''
    current_column = 0
    wait_for_nl = False
    for char in prefix:
        current_line += char
        if wait_for_nl:
            if char == '\n':
                if current_line.strip() and current_column < column:
                    res = ''.join(lines)
                    return (res, prefix[len(res):])
                lines.append(current_line)
                current_line = ''
                current_column = 0
                wait_for_nl = False
        elif char in ' \t':
            current_column += 1
        elif char == '\n':
            current_column = 0
        elif char == '\x0c':
            current_column = 0
        else:
            wait_for_nl = True
    return (''.join(lines), current_line)