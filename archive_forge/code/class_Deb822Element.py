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
class Deb822Element:
    """Composite elements (consists of 1 or more tokens)"""
    __slots__ = ('_parent_element', '__weakref__')

    def __init__(self):
        self._parent_element = None

    def iter_parts(self):
        raise NotImplementedError

    def iter_parts_of_type(self, only_element_or_token_type):
        for part in self.iter_parts():
            if isinstance(part, only_element_or_token_type):
                yield part

    def iter_tokens(self):
        for part in self.iter_parts():
            assert part._parent_element is not None
            if isinstance(part, Deb822Element):
                yield from part.iter_tokens()
            else:
                yield part

    def iter_recurse(self, *, only_element_or_token_type=None):
        for part in self.iter_parts():
            if only_element_or_token_type is None or isinstance(part, only_element_or_token_type):
                yield cast('TE', part)
            if isinstance(part, Deb822Element):
                yield from part.iter_recurse(only_element_or_token_type=only_element_or_token_type)

    @property
    def parent_element(self):
        return resolve_ref(self._parent_element)

    @parent_element.setter
    def parent_element(self, new_parent):
        self._parent_element = weakref.ref(new_parent) if new_parent is not None else None

    def _init_parent_of_parts(self):
        for part in self.iter_parts():
            part.parent_element = self

    def convert_to_text(self):
        return ''.join((t.text for t in self.iter_tokens()))

    def clear_parent_if_parent(self, parent):
        if parent is self.parent_element:
            self._parent_element = None