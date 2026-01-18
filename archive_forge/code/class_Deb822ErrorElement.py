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
class Deb822ErrorElement(Deb822Element):
    """Element representing elements or tokens that are out of place

    Commonly, it will just be instances of Deb822ErrorToken, but it can be other
    things.  As an example if a parser discovers out of order elements/tokens,
    it can bundle them in a Deb822ErrorElement to signal that the sequence of
    elements/tokens are invalid (even if the tokens themselves are valid).
    """
    __slots__ = ('_parts',)

    def __init__(self, parts):
        super().__init__()
        self._parts = parts
        self._init_parent_of_parts()

    def iter_parts(self):
        yield from self._parts