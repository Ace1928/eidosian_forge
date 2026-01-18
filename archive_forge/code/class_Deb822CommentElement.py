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
class Deb822CommentElement(Deb822Element):
    __slots__ = ('_comment_tokens',)

    def __init__(self, comment_tokens):
        super().__init__()
        self._comment_tokens = comment_tokens
        if not comment_tokens:
            raise ValueError('Comment elements must have at least one comment token')
        self._init_parent_of_parts()

    def __len__(self):
        return len(self._comment_tokens)

    def __getitem__(self, item):
        return self._comment_tokens[item]

    def iter_parts(self):
        yield from self._comment_tokens