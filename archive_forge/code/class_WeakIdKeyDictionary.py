from __future__ import annotations
import weakref
from weakref import ref
from _weakrefset import _IterationGuard  # type: ignore[attr-defined]
from collections.abc import MutableMapping, Mapping
from torch import Tensor
import collections.abc as _collections_abc
class WeakIdKeyDictionary(MutableMapping):
    data: dict[WeakIdRef, object]

    def __init__(self, dict=None):
        self.data = {}

        def remove(k, selfref=ref(self)):
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(k)
                else:
                    try:
                        del self.data[k]
                    except KeyError:
                        pass
        self._remove = remove
        self._pending_removals = []
        self._iterating = set()
        self._dirty_len = False
        if dict is not None:
            self.update(dict)

    def _commit_removals(self):
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return
            try:
                del d[key]
            except KeyError:
                pass

    def _scrub_removals(self):
        d = self.data
        self._pending_removals = [k for k in self._pending_removals if k in d]
        self._dirty_len = False

    def __delitem__(self, key):
        self._dirty_len = True
        del self.data[WeakIdRef(key)]

    def __getitem__(self, key):
        return self.data[WeakIdRef(key)]

    def __len__(self):
        if self._dirty_len and self._pending_removals:
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self):#x}>'

    def __setitem__(self, key, value):
        self.data[WeakIdRef(key, self._remove)] = value

    def copy(self):
        new = WeakIdKeyDictionary()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = value
        return new
    __copy__ = copy

    def __deepcopy__(self, memo):
        from copy import deepcopy
        new = self.__class__()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = deepcopy(value, memo)
        return new

    def get(self, key, default=None):
        return self.data.get(WeakIdRef(key), default)

    def __contains__(self, key):
        try:
            wr = WeakIdRef(key)
        except TypeError:
            return False
        return wr in self.data

    def items(self):
        with _IterationGuard(self):
            for wr, value in self.data.items():
                key = wr()
                if key is not None:
                    yield (key, value)

    def keys(self):
        with _IterationGuard(self):
            for wr in self.data:
                obj = wr()
                if obj is not None:
                    yield obj
    __iter__ = keys

    def values(self):
        with _IterationGuard(self):
            for wr, value in self.data.items():
                if wr() is not None:
                    yield value

    def keyrefs(self):
        """Return a list of weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
        return list(self.data)

    def popitem(self):
        self._dirty_len = True
        while True:
            key, value = self.data.popitem()
            o = key()
            if o is not None:
                return (o, value)

    def pop(self, key, *args):
        self._dirty_len = True
        return self.data.pop(WeakIdRef(key), *args)

    def setdefault(self, key, default=None):
        return self.data.setdefault(WeakIdRef(key, self._remove), default)

    def update(self, dict=None, **kwargs):
        d = self.data
        if dict is not None:
            if not hasattr(dict, 'items'):
                dict = type({})(dict)
            for key, value in dict.items():
                d[WeakIdRef(key, self._remove)] = value
        if len(kwargs):
            self.update(kwargs)

    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return {id(k): v for k, v in self.items()} == {id(k): v for k, v in other.items()}