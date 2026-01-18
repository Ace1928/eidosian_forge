import functools
import itertools
import operator
import warnings
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx
def _vhstack_dispatcher(tup, *, dtype=None, casting=None):
    return _arrays_for_stack_dispatcher(tup)