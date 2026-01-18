from __future__ import annotations
import inspect
import random
import threading
from collections import OrderedDict, UserDict
from collections.abc import Iterable, Mapping
from itertools import count, repeat
from time import sleep, time
from vine.utils import wraps
from .encoding import safe_repr as _safe_repr
@wraps(fun)
def _M(*args, **kwargs):
    if keyfun:
        key = keyfun(args, kwargs)
    else:
        key = args + (KEYWORD_MARK,) + tuple(sorted(kwargs.items()))
    try:
        with mutex:
            value = cache[key]
    except KeyError:
        value = fun(*args, **kwargs)
        _M.misses += 1
        with mutex:
            cache[key] = value
    else:
        _M.hits += 1
    return value