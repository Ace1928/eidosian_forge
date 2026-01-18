import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def _make_getstate_setstate(self):
    """
        Create custom __setstate__ and __getstate__ methods.
        """
    state_attr_names = tuple((an for an in self._attr_names if an != '__weakref__'))

    def slots_getstate(self):
        """
            Automatically created by attrs.
            """
        return {name: getattr(self, name) for name in state_attr_names}
    hash_caching_enabled = self._cache_hash

    def slots_setstate(self, state):
        """
            Automatically created by attrs.
            """
        __bound_setattr = _obj_setattr.__get__(self)
        if isinstance(state, tuple):
            for name, value in zip(state_attr_names, state):
                __bound_setattr(name, value)
        else:
            for name in state_attr_names:
                if name in state:
                    __bound_setattr(name, state[name])
        if hash_caching_enabled:
            __bound_setattr(_hash_cache_field, None)
    return (slots_getstate, slots_setstate)