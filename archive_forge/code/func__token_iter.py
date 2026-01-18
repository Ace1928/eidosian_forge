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