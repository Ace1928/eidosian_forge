import inspect
import warnings
import types
import collections
import itertools
from functools import lru_cache, wraps
from typing import Callable, List, Union, Iterable, TypeVar, cast
def _escape_regex_range_chars(s: str) -> str:
    for c in '\\^-[]':
        s = s.replace(c, _bslash + c)
    s = s.replace('\n', '\\n')
    s = s.replace('\t', '\\t')
    return str(s)