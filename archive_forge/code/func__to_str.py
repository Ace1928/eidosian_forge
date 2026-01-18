from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _to_str(s, encoding='utf8', errors='ignore'):
    """
    A type safe wrapper for converting a bytestring to str. This is essentially just
    a wrapper around .decode() intended for use with things like map(), but with some
    specific behavior:

    1. if the given parameter is not a bytestring, it is returned unmodified
    2. decode() is called for the given parameter and assumes utf8 encoding, but the
       default error behavior is changed from 'strict' to 'ignore'

    >>> repr(_to_str(b'foo'))
    "'foo'"

    >>> repr(_to_str('foo'))
    "'foo'"

    >>> repr(_to_str(42))
    "'42'"

    """
    if isinstance(s, bytes):
        return s.decode(encoding=encoding, errors=errors)
    return str(s)