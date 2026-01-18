from collections.abc import Mapping, Hashable
from itertools import chain
from typing import Generic, TypeVar
from pyrsistent._pvector import pvector
from pyrsistent._transformations import transform
class _Evolver(object):
    __slots__ = ('_buckets_evolver', '_size', '_original_pmap')

    def __init__(self, original_pmap):
        self._original_pmap = original_pmap
        self._buckets_evolver = original_pmap._buckets.evolver()
        self._size = original_pmap._size

    def __getitem__(self, key):
        return PMap._getitem(self._buckets_evolver, key)

    def __setitem__(self, key, val):
        self.set(key, val)

    def set(self, key, val):
        kv = (key, val)
        index, bucket = PMap._get_bucket(self._buckets_evolver, key)
        reallocation_required = len(self._buckets_evolver) < 0.67 * self._size
        if bucket:
            for k, v in bucket:
                if k == key:
                    if v is not val:
                        new_bucket = [(k2, v2) if not k2 == k else (k2, val) for k2, v2 in bucket]
                        self._buckets_evolver[index] = new_bucket
                    return self
            if reallocation_required:
                self._reallocate()
                return self.set(key, val)
            new_bucket = [kv]
            new_bucket.extend(bucket)
            self._buckets_evolver[index] = new_bucket
            self._size += 1
        else:
            if reallocation_required:
                self._reallocate()
                return self.set(key, val)
            self._buckets_evolver[index] = [kv]
            self._size += 1
        return self

    def _reallocate(self):
        new_size = 2 * len(self._buckets_evolver)
        new_list = new_size * [None]
        buckets = self._buckets_evolver.persistent()
        for k, v in chain.from_iterable((x for x in buckets if x)):
            index = hash(k) % new_size
            if new_list[index]:
                new_list[index].append((k, v))
            else:
                new_list[index] = [(k, v)]
        self._buckets_evolver = pvector().evolver()
        self._buckets_evolver.extend(new_list)

    def is_dirty(self):
        return self._buckets_evolver.is_dirty()

    def persistent(self):
        if self.is_dirty():
            self._original_pmap = PMap(self._size, self._buckets_evolver.persistent())
        return self._original_pmap

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return PMap._contains(self._buckets_evolver, key)

    def __delitem__(self, key):
        self.remove(key)

    def remove(self, key):
        index, bucket = PMap._get_bucket(self._buckets_evolver, key)
        if bucket:
            new_bucket = [(k, v) for k, v in bucket if not k == key]
            size_diff = len(bucket) - len(new_bucket)
            if size_diff > 0:
                self._buckets_evolver[index] = new_bucket if new_bucket else None
                self._size -= size_diff
                return self
        raise KeyError('{0}'.format(key))