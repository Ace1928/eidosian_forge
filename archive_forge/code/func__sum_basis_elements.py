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
def _sum_basis_elements(x, t, c, k):
    n = len(t) - (k + 1)
    assert n >= k + 1
    assert len(c) >= n
    s = 0.0
    for i in range(n):
        b = BSpline.basis_element(t[i:i + k + 2], extrapolate=False)(x)
        s += c[i] * np.nan_to_num(b)
    return s