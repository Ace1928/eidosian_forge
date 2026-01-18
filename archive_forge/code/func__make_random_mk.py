from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
def _make_random_mk(self, m, k):
    np.random.seed(1234)
    xi = np.asarray([1.0 * j ** 2 for j in range(m + 1)])
    yi = [np.random.random(k) for j in range(m + 1)]
    return (xi, yi)