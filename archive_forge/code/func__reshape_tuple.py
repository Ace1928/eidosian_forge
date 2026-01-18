import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@lazy_cache('reshape')
def _reshape_tuple(a, newshape):
    a = ensure_lazy(a)
    fn_reshape = get_lib_fn(a.backend, 'reshape')
    if a._fn is fn_reshape:
        b = a._args[0]
        if isinstance(b, LazyArray):
            a = b
    return a.to(fn_reshape, (a, newshape), shape=newshape)