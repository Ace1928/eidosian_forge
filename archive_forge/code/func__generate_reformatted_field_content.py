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
def _generate_reformatted_field_content(self):
    separator_token = self._default_separator_factory()
    vtype = self._vtype
    stype = self._stype
    token_list = self._token_list

    def _token_iter():
        text = ''
        for te in token_list:
            if isinstance(te, Deb822Token):
                if te.is_comment:
                    yield FormatterContentToken.comment_token(te.text)
                elif isinstance(te, stype):
                    text = te.text
                    yield FormatterContentToken.separator_token(text)
            else:
                assert isinstance(te, vtype)
                text = te.convert_to_text()
                yield FormatterContentToken.value_token(text)
    return format_field(self._formatter, self._kvpair_element.field_name, FormatterContentToken.separator_token(separator_token.text), _token_iter())