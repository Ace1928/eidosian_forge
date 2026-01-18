from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
class DtypeMapper(ty.Dict[ty.Hashable, ty.Hashable]):
    """Specialized mapper for numpy dtypes

    We pass this mapper into the Recoder class to deal with numpy dtype
    hashing.

    The hashing problem is that dtypes that compare equal may not have the same
    hash.  This is true for numpys up to the current at time of writing
    (1.6.0).  For numpy 1.2.1 at least, even dtypes that look exactly the same
    in terms of fields don't always have the same hash.  This makes dtypes
    difficult to use as keys in a dictionary.

    This class wraps a dictionary in order to implement a __getitem__ to deal
    with dtype hashing. If the key doesn't appear to be in the mapping, and it
    is a dtype, we compare (using ==) all known dtype keys to the input key,
    and return any matching values for the matching key.
    """

    def __init__(self) -> None:
        super().__init__()
        self._dtype_keys: list[np.dtype] = []

    def __setitem__(self, key: ty.Hashable, value: ty.Hashable) -> None:
        """Set item into mapping, checking for dtype keys

        Cache dtype keys for comparison test in __getitem__
        """
        super().__setitem__(key, value)
        if isinstance(key, np.dtype):
            self._dtype_keys.append(key)

    def __getitem__(self, key: ty.Hashable) -> ty.Hashable:
        """Get item from mapping, checking for dtype keys

        First do simple hash lookup, then check for a dtype key that has failed
        the hash lookup.  Look then for any known dtype keys that compare equal
        to `key`.
        """
        try:
            return super().__getitem__(key)
        except KeyError:
            pass
        if isinstance(key, np.dtype):
            for dt in self._dtype_keys:
                if key == dt:
                    return super().__getitem__(dt)
        raise KeyError(key)