from itertools import chain
from operator import eq, ne, gt, ge, lt, le
from textwrap import dedent
from .sortedlist import SortedList, recursive_repr
@classmethod
def _fromset(cls, values, key=None):
    """Initialize sorted set from existing set.

        Used internally by set operations that return a new set.

        """
    sorted_set = object.__new__(cls)
    sorted_set._set = values
    sorted_set.__init__(key=key)
    return sorted_set