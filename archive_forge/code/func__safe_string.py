import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def _safe_string(value, what, func=str):
    try:
        return func(value)
    except:
        return f'<{what} {func.__name__}() failed>'