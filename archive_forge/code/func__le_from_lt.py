from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def _le_from_lt(self, other):
    """Return a <= b.  Computed by @total_ordering from (a < b) or (a == b)."""
    op_result = type(self).__lt__(self, other)
    if op_result is NotImplemented:
        return op_result
    return op_result or self == other