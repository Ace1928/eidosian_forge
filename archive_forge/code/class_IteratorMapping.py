import itertools
from .compat import collections_abc
class IteratorMapping(collections_abc.Mapping):

    def __init__(self, mapping, accessor, appends=None):
        self._mapping = mapping
        self._accessor = accessor
        self._appends = appends or {}

    def __repr__(self):
        return 'IteratorMapping({!r}, {!r}, {!r})'.format(self._mapping, self._accessor, self._appends)

    def __bool__(self):
        return bool(self._mapping or self._appends)
    __nonzero__ = __bool__

    def __contains__(self, key):
        return key in self._mapping or key in self._appends

    def __getitem__(self, k):
        try:
            v = self._mapping[k]
        except KeyError:
            return iter(self._appends[k])
        return itertools.chain(self._accessor(v), self._appends.get(k, ()))

    def __iter__(self):
        more = (k for k in self._appends if k not in self._mapping)
        return itertools.chain(self._mapping, more)

    def __len__(self):
        more = sum((1 for k in self._appends if k not in self._mapping))
        return len(self._mapping) + more