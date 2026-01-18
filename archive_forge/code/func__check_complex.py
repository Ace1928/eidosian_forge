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
def _check_complex(self, dtype=np.complex128, kind='linear'):
    x = np.array([1, 2.5, 3, 3.1, 4, 6.4, 7.9, 8.0, 9.5, 10])
    y = x * x ** (1 + 2j)
    y = y.astype(dtype)
    c = interp1d(x, y, kind=kind)
    assert_array_almost_equal(y[:-1], c(x)[:-1])
    xi = np.linspace(1, 10, 31)
    cr = interp1d(x, y.real, kind=kind)
    ci = interp1d(x, y.imag, kind=kind)
    assert_array_almost_equal(c(xi).real, cr(xi))
    assert_array_almost_equal(c(xi).imag, ci(xi))