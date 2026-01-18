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
def _make_multiples(b):
    """Increase knot multiplicity."""
    c, k = (b.c, b.k)
    t1 = b.t.copy()
    t1[17:19] = t1[17]
    t1[22] = t1[21]
    yield BSpline(t1, c, k)
    t1 = b.t.copy()
    t1[:k + 1] = t1[0]
    yield BSpline(t1, c, k)
    t1 = b.t.copy()
    t1[-k - 1:] = t1[-1]
    yield BSpline(t1, c, k)