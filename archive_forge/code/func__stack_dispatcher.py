import functools
import itertools
import operator
import warnings
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx
def _stack_dispatcher(arrays, axis=None, out=None, *, dtype=None, casting=None):
    arrays = _arrays_for_stack_dispatcher(arrays)
    if out is not None:
        arrays = list(arrays)
        arrays.append(out)
    return arrays