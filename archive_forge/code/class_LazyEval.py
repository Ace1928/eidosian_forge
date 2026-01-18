from __future__ import absolute_import, print_function
from functools import partial
import re
from .compat import text_type, binary_type
class LazyEval(object):
    """
    Lightweight wrapper around lazily evaluated func(*args, **kwargs).

    func is only evaluated when any attribute of its return value is accessed.
    Every attribute access is passed through to the wrapped value.
    (This only excludes special cases like method-wrappers, e.g., __hash__.)
    The sole additional attribute is the lazy_self function which holds the
    return value (or, prior to evaluation, func and arguments), in its closure.
    """

    def __init__(self, func, *args, **kwargs):

        def lazy_self():
            return_value = func(*args, **kwargs)
            object.__setattr__(self, 'lazy_self', lambda: return_value)
            return return_value
        object.__setattr__(self, 'lazy_self', lazy_self)

    def __getattribute__(self, name):
        lazy_self = object.__getattribute__(self, 'lazy_self')
        if name == 'lazy_self':
            return lazy_self
        return getattr(lazy_self(), name)

    def __setattr__(self, name, value):
        setattr(self.lazy_self(), name, value)