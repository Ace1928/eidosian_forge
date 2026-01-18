from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def _c3_merge(sequences):
    """Merges MROs in *sequences* to a single MRO using the C3 algorithm.

    Adapted from https://www.python.org/download/releases/2.3/mro/.

    """
    result = []
    while True:
        sequences = [s for s in sequences if s]
        if not sequences:
            return result
        for s1 in sequences:
            candidate = s1[0]
            for s2 in sequences:
                if candidate in s2[1:]:
                    candidate = None
                    break
            else:
                break
        if candidate is None:
            raise RuntimeError('Inconsistent hierarchy')
        result.append(candidate)
        for seq in sequences:
            if seq[0] == candidate:
                del seq[0]