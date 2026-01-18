import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
def _naive_B(x, k, i, t):
    """
    Naive way to compute B-spline basis functions. Useful only for testing!
    computes B(x; t[i],..., t[i+k+1])
    """
    if k == 0:
        return 1.0 if t[i] <= x < t[i + 1] else 0.0
    if t[i + k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i]) / (t[i + k] - t[i]) * _naive_B(x, k - 1, i, t)
    if t[i + k + 1] == t[i + 1]:
        c2 = 0.0
    else:
        c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * _naive_B(x, k - 1, i + 1, t)
    return c1 + c2