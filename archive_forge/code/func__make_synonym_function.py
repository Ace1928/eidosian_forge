import inspect
import warnings
import types
import collections
import itertools
from functools import lru_cache, wraps
from typing import Callable, List, Union, Iterable, TypeVar, cast
def _make_synonym_function(compat_name: str, fn: C) -> C:
    fn = getattr(fn, '__func__', fn)
    if 'self' == list(inspect.signature(fn).parameters)[0]:

        @wraps(fn)
        def _inner(self, *args, **kwargs):
            return fn(self, *args, **kwargs)
    else:

        @wraps(fn)
        def _inner(*args, **kwargs):
            return fn(*args, **kwargs)
    _inner.__doc__ = f'Deprecated - use :class:`{fn.__name__}`'
    _inner.__name__ = compat_name
    _inner.__annotations__ = fn.__annotations__
    if isinstance(fn, types.FunctionType):
        _inner.__kwdefaults__ = fn.__kwdefaults__
    elif isinstance(fn, type) and hasattr(fn, '__init__'):
        _inner.__kwdefaults__ = fn.__init__.__kwdefaults__
    else:
        _inner.__kwdefaults__ = None
    _inner.__qualname__ = fn.__qualname__
    return cast(C, _inner)