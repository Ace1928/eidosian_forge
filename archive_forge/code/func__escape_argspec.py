import functools
import re
import string
import sys
import typing as t
def _escape_argspec(obj: _ListOrDict, iterable: t.Iterable[t.Any], escape: t.Callable[[t.Any], Markup]) -> _ListOrDict:
    """Helper for various string-wrapped functions."""
    for key, value in iterable:
        if isinstance(value, str) or hasattr(value, '__html__'):
            obj[key] = escape(value)
    return obj