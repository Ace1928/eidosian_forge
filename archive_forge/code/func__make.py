import _collections_abc
import sys as _sys
from itertools import chain as _chain
from itertools import repeat as _repeat
from itertools import starmap as _starmap
from keyword import iskeyword as _iskeyword
from operator import eq as _eq
from operator import itemgetter as _itemgetter
from reprlib import recursive_repr as _recursive_repr
from _weakref import proxy as _proxy
@classmethod
def _make(cls, iterable):
    result = tuple_new(cls, iterable)
    if _len(result) != num_fields:
        raise TypeError(f'Expected {num_fields} arguments, got {len(result)}')
    return result