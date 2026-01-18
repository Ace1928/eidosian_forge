from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
def _check_arrays_broadcastable(arrays, axis):
    n_dims = max([arr.ndim for arr in arrays])
    if axis is not None:
        axis = -n_dims + axis if axis >= 0 else axis
    for dim in range(1, n_dims + 1):
        if -dim == axis:
            continue
        dim_lengths = set()
        for arr in arrays:
            if dim <= arr.ndim and arr.shape[-dim] != 1:
                dim_lengths.add(arr.shape[-dim])
        if len(dim_lengths) > 1:
            return False
    return True