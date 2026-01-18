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
class Deb822DictishParagraphWrapper(AbstractDeb822ParagraphWrapper[str], Deb822ParagraphToStrWrapperMixin):

    def __init__(self, paragraph, *, discard_comments_on_read=True, auto_map_initial_line_whitespace=True, auto_resolve_ambiguous_fields=False, preserve_field_comments_on_field_updates=True, auto_map_final_newline_in_multiline_values=True):
        super().__init__(paragraph, auto_resolve_ambiguous_fields=auto_resolve_ambiguous_fields, discard_comments_on_read=discard_comments_on_read)
        self.__auto_map_initial_line_whitespace = auto_map_initial_line_whitespace
        self.__preserve_field_comments_on_field_updates = preserve_field_comments_on_field_updates
        self.__auto_map_final_newline_in_multiline_values = auto_map_final_newline_in_multiline_values

    @property
    def _auto_map_initial_line_whitespace(self):
        return self.__auto_map_initial_line_whitespace

    @property
    def _preserve_field_comments_on_field_updates(self):
        return self.__preserve_field_comments_on_field_updates

    @property
    def _auto_map_final_newline_in_multiline_values(self):
        return self.__auto_map_final_newline_in_multiline_values