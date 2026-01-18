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
def _format_comment(c):
    if c == '':
        return '#\n'
    if '\n' in c[:-1]:
        raise ValueError('Comment lines must not have embedded newlines')
    if not c.endswith('\n'):
        c = c.rstrip() + '\n'
    if not c.startswith('#'):
        c = '# ' + c.lstrip()
    return c