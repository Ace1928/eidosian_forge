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
@property
def _continuation_line_char(self):
    char = self.__continuation_line_char
    if char is None:
        char = ' '
        for token in self._token_list:
            if isinstance(token, Deb822ValueContinuationToken):
                char = token.text
                break
        self.__continuation_line_char = char
    return char