import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def _parse_value_tb(exc, value, tb):
    if (value is _sentinel) != (tb is _sentinel):
        raise ValueError('Both or neither of value and tb must be given')
    if value is tb is _sentinel:
        if exc is not None:
            if isinstance(exc, BaseException):
                return (exc, exc.__traceback__)
            raise TypeError(f'Exception expected for value, {type(exc).__name__} found')
        else:
            return (None, None)
    return (value, tb)