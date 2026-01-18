import functools
import operator
from numpy.core.numeric import (
from numpy.core.overrides import set_array_function_like_doc, set_module
from numpy.core import overrides
from numpy.core import iinfo
from numpy.lib.stride_tricks import broadcast_to
def _histogram2d_dispatcher(x, y, bins=None, range=None, density=None, weights=None):
    yield x
    yield y
    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N == 2:
        yield from bins
    else:
        yield bins
    yield weights