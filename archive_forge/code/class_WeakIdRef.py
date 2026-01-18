from __future__ import annotations
import weakref
from weakref import ref
from _weakrefset import _IterationGuard  # type: ignore[attr-defined]
from collections.abc import MutableMapping, Mapping
from torch import Tensor
import collections.abc as _collections_abc
class WeakIdRef(weakref.ref):
    __slots__ = ['_id']

    def __init__(self, key, callback=None):
        self._id = id(key)
        super().__init__(key, callback)

    def __call__(self):
        r = super().__call__()
        if hasattr(r, '_fix_weakref'):
            r._fix_weakref()
        return r

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        a = self()
        b = other()
        if a is not None and b is not None:
            return a is b
        return self is other