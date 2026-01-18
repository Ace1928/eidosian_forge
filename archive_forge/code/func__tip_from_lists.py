from collections.abc import Sequence, Hashable
from itertools import islice, chain
from numbers import Integral
from typing import TypeVar, Generic
from pyrsistent._plist import plist
@staticmethod
def _tip_from_lists(primary_list, secondary_list):
    if primary_list:
        return primary_list.first
    if secondary_list:
        return secondary_list[-1]
    raise IndexError('No elements in empty deque')