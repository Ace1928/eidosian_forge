import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def _parse_inner():
    string_start = 0
    string_end = 0
    in_fmt = False
    i = 0
    while i < len(s):
        if not in_fmt:
            try:
                i = s.index('%', i)
            except ValueError:
                yield (s[string_start:], None)
                return
            else:
                string_end = i
                i += 1
                in_fmt = True
        else:
            key_match = MAPPING_KEY_RE.match(s, i)
            if key_match:
                key = key_match.group(1)
                i = key_match.end()
            else:
                key = None
            conversion_flag_match = _must_match(CONVERSION_FLAG_RE, s, i)
            conversion_flag = conversion_flag_match.group() or None
            i = conversion_flag_match.end()
            width_match = _must_match(WIDTH_RE, s, i)
            width = width_match.group() or None
            i = width_match.end()
            precision_match = _must_match(PRECISION_RE, s, i)
            precision = precision_match.group() or None
            i = precision_match.end()
            i = _must_match(LENGTH_RE, s, i).end()
            try:
                conversion = s[i]
            except IndexError:
                raise ValueError('end-of-string while parsing format')
            i += 1
            fmt = (key, conversion_flag, width, precision, conversion)
            yield (s[string_start:string_end], fmt)
            in_fmt = False
            string_start = i
    if in_fmt:
        raise ValueError('end-of-string while parsing format')