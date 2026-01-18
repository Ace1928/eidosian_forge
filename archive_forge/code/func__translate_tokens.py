import abc
import ast
import bisect
import sys
import token
from ast import Module
from typing import Iterable, Iterator, List, Optional, Tuple, Any, cast, TYPE_CHECKING
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from .line_numbers import LineNumbers
from .util import (
def _translate_tokens(self, original_tokens):
    """
    Translates the given standard library tokens into our own representation.
    """
    for index, tok in enumerate(patched_generate_tokens(original_tokens)):
        tok_type, tok_str, start, end, line = tok
        yield Token(tok_type, tok_str, start, end, line, index, self._line_numbers.line_to_offset(start[0], start[1]), self._line_numbers.line_to_offset(end[0], end[1]))