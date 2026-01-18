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
class Deb822ValueLineElement(Deb822Element):
    """Consists of one "line" of a value"""
    __slots__ = ('_comment_element', '_continuation_line_token', '_leading_whitespace_token', '_value_tokens', '_trailing_whitespace_token', '_newline_token')

    def __init__(self, comment_element, continuation_line_token, leading_whitespace_token, value_parts, trailing_whitespace_token, newline_token):
        super().__init__()
        if comment_element is not None and continuation_line_token is None:
            raise ValueError('Only continuation lines can have comments')
        self._comment_element = comment_element
        self._continuation_line_token = continuation_line_token
        self._leading_whitespace_token = leading_whitespace_token
        self._value_tokens = value_parts
        self._trailing_whitespace_token = trailing_whitespace_token
        self._newline_token = newline_token
        self._init_parent_of_parts()

    @property
    def comment_element(self):
        return self._comment_element

    @property
    def continuation_line_token(self):
        return self._continuation_line_token

    @property
    def newline_token(self):
        return self._newline_token

    def add_newline_if_missing(self):
        if self._newline_token is None:
            self._newline_token = Deb822NewlineAfterValueToken()
            self._newline_token.parent_element = self

    def _iter_content_parts(self):
        if self._leading_whitespace_token:
            yield self._leading_whitespace_token
        yield from self._value_tokens
        if self._trailing_whitespace_token:
            yield self._trailing_whitespace_token

    def _iter_content_tokens(self):
        for part in self._iter_content_parts():
            if isinstance(part, Deb822Element):
                yield from part.iter_tokens()
            else:
                yield part

    def convert_content_to_text(self):
        if len(self._value_tokens) == 1 and (not self._leading_whitespace_token) and (not self._trailing_whitespace_token) and isinstance(self._value_tokens[0], Deb822Token):
            return self._value_tokens[0].text
        return ''.join((t.text for t in self._iter_content_tokens()))

    def iter_parts(self):
        if self._comment_element:
            yield self._comment_element
        if self._continuation_line_token:
            yield self._continuation_line_token
        yield from self._iter_content_parts()
        if self._newline_token:
            yield self._newline_token