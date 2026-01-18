import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
def _resolve_node(self):
    if self._node is None:
        raise RuntimeError('Cannot use ValueReference after remove()')
    node = self._node()
    if node is None:
        raise RuntimeError('ValueReference is invalid (garbage collected)')
    return node