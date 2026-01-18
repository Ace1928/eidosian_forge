import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def _weight_masked(arrays, weights, axis):
    if axis is None:
        axis = 0
    weights = np.asanyarray(weights)
    for a in arrays:
        axis_mask = np.ma.getmask(a)
        if axis_mask is np.ma.nomask:
            continue
        if a.ndim > 1:
            not_axes = tuple((i for i in range(a.ndim) if i != axis))
            axis_mask = axis_mask.any(axis=not_axes)
        weights *= 1 - axis_mask.astype(int)
    return weights