from collections.abc import Mapping, Hashable
from itertools import chain
from typing import Generic, TypeVar
from pyrsistent._pvector import pvector
from pyrsistent._transformations import transform
def _turbo_mapping(initial, pre_size):
    if pre_size:
        size = pre_size
    else:
        try:
            size = 2 * len(initial) or 8
        except Exception:
            size = 8
    buckets = size * [None]
    if not isinstance(initial, Mapping):
        initial = dict(initial)
    for k, v in initial.items():
        h = hash(k)
        index = h % size
        bucket = buckets[index]
        if bucket:
            bucket.append((k, v))
        else:
            buckets[index] = [(k, v)]
    return PMap(len(initial), pvector().extend(buckets))