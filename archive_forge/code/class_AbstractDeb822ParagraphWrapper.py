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
class AbstractDeb822ParagraphWrapper(AutoResolvingMixin[T], ABC):

    def __init__(self, paragraph, *, auto_resolve_ambiguous_fields=False, discard_comments_on_read=True):
        self.__paragraph = paragraph
        self.__auto_resolve_ambiguous_fields = auto_resolve_ambiguous_fields
        self.__discard_comments_on_read = discard_comments_on_read

    @property
    def _paragraph(self):
        return self.__paragraph

    @property
    def _discard_comments_on_read(self):
        return self.__discard_comments_on_read

    @property
    def _auto_resolve_ambiguous_fields(self):
        return self.__auto_resolve_ambiguous_fields