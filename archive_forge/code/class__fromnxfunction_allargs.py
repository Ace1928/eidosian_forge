import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
class _fromnxfunction_allargs(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with multiple array
    arguments. Similar to `_fromnxfunction_args` except that all args
    are converted to arrays even if they are not so already. This makes
    it possible to process scalars as 1-D arrays. Only keyword arguments
    are passed through verbatim for the data and mask calls. Arrays
    arguments are processed independently and the results are returned
    in a list. If only one arg is present, the return value is just the
    processed array instead of a list.
    """

    def __call__(self, *args, **params):
        func = getattr(np, self.__name__)
        res = []
        for x in args:
            _d = func(np.asarray(x), **params)
            _m = func(getmaskarray(x), **params)
            res.append(masked_array(_d, mask=_m))
        if len(args) == 1:
            return res[0]
        return res