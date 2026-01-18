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
def _afterpoint(string):
    """Symbols after a decimal point, -1 if the string lacks the decimal point.

    >>> _afterpoint("123.45")
    2
    >>> _afterpoint("1001")
    -1
    >>> _afterpoint("eggs")
    -1
    >>> _afterpoint("123e45")
    2
    >>> _afterpoint("123,456.78")
    2

    """
    if _isnumber(string) or _isnumber_with_thousands_separator(string):
        if _isint(string):
            return -1
        else:
            pos = string.rfind('.')
            pos = string.lower().rfind('e') if pos < 0 else pos
            if pos >= 0:
                return len(string) - pos - 1
            else:
                return -1
    else:
        return -1