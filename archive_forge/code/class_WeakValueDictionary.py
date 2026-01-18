from _weakref import (
from _weakrefset import WeakSet, _IterationGuard
import _collections_abc  # Import after _weakref to avoid circular import.
import sys
import itertools
class WeakValueDictionary(_collections_abc.MutableMapping):
    """Mapping class that references values weakly.

    Entries in the dictionary will be discarded when no strong
    reference to the value exists anymore
    """

    def __init__(self, other=(), /, **kw):

        def remove(wr, selfref=ref(self), _atomic_removal=_remove_dead_weakref):
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(wr.key)
                else:
                    _atomic_removal(self.data, wr.key)
        self._remove = remove
        self._pending_removals = []
        self._iterating = set()
        self.data = {}
        self.update(other, **kw)

    def _commit_removals(self, _atomic_removal=_remove_dead_weakref):
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return
            _atomic_removal(d, key)

    def __getitem__(self, key):
        if self._pending_removals:
            self._commit_removals()
        o = self.data[key]()
        if o is None:
            raise KeyError(key)
        else:
            return o

    def __delitem__(self, key):
        if self._pending_removals:
            self._commit_removals()
        del self.data[key]

    def __len__(self):
        if self._pending_removals:
            self._commit_removals()
        return len(self.data)

    def __contains__(self, key):
        if self._pending_removals:
            self._commit_removals()
        try:
            o = self.data[key]()
        except KeyError:
            return False
        return o is not None

    def __repr__(self):
        return '<%s at %#x>' % (self.__class__.__name__, id(self))

    def __setitem__(self, key, value):
        if self._pending_removals:
            self._commit_removals()
        self.data[key] = KeyedRef(value, self._remove, key)

    def copy(self):
        if self._pending_removals:
            self._commit_removals()
        new = WeakValueDictionary()
        with _IterationGuard(self):
            for key, wr in self.data.items():
                o = wr()
                if o is not None:
                    new[key] = o
        return new
    __copy__ = copy

    def __deepcopy__(self, memo):
        from copy import deepcopy
        if self._pending_removals:
            self._commit_removals()
        new = self.__class__()
        with _IterationGuard(self):
            for key, wr in self.data.items():
                o = wr()
                if o is not None:
                    new[deepcopy(key, memo)] = o
        return new

    def get(self, key, default=None):
        if self._pending_removals:
            self._commit_removals()
        try:
            wr = self.data[key]
        except KeyError:
            return default
        else:
            o = wr()
            if o is None:
                return default
            else:
                return o

    def items(self):
        if self._pending_removals:
            self._commit_removals()
        with _IterationGuard(self):
            for k, wr in self.data.items():
                v = wr()
                if v is not None:
                    yield (k, v)

    def keys(self):
        if self._pending_removals:
            self._commit_removals()
        with _IterationGuard(self):
            for k, wr in self.data.items():
                if wr() is not None:
                    yield k
    __iter__ = keys

    def itervaluerefs(self):
        """Return an iterator that yields the weak references to the values.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the values around longer than needed.

        """
        if self._pending_removals:
            self._commit_removals()
        with _IterationGuard(self):
            yield from self.data.values()

    def values(self):
        if self._pending_removals:
            self._commit_removals()
        with _IterationGuard(self):
            for wr in self.data.values():
                obj = wr()
                if obj is not None:
                    yield obj

    def popitem(self):
        if self._pending_removals:
            self._commit_removals()
        while True:
            key, wr = self.data.popitem()
            o = wr()
            if o is not None:
                return (key, o)

    def pop(self, key, *args):
        if self._pending_removals:
            self._commit_removals()
        try:
            o = self.data.pop(key)()
        except KeyError:
            o = None
        if o is None:
            if args:
                return args[0]
            else:
                raise KeyError(key)
        else:
            return o

    def setdefault(self, key, default=None):
        try:
            o = self.data[key]()
        except KeyError:
            o = None
        if o is None:
            if self._pending_removals:
                self._commit_removals()
            self.data[key] = KeyedRef(default, self._remove, key)
            return default
        else:
            return o

    def update(self, other=None, /, **kwargs):
        if self._pending_removals:
            self._commit_removals()
        d = self.data
        if other is not None:
            if not hasattr(other, 'items'):
                other = dict(other)
            for key, o in other.items():
                d[key] = KeyedRef(o, self._remove, key)
        for key, o in kwargs.items():
            d[key] = KeyedRef(o, self._remove, key)

    def valuerefs(self):
        """Return a list of weak references to the values.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the values around longer than needed.

        """
        if self._pending_removals:
            self._commit_removals()
        return list(self.data.values())

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