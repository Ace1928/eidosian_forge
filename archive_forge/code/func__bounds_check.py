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
def _bounds_check(self, kind='linear'):
    extrap10 = interp1d(self.x10, self.y10, fill_value=self.fill_value, bounds_error=False, kind=kind)
    assert_array_equal(extrap10(11.2), np.array(self.fill_value))
    assert_array_equal(extrap10(-3.4), np.array(self.fill_value))
    assert_array_equal(extrap10([[[11.2], [-3.4], [12.6], [19.3]]]), np.array(self.fill_value))
    assert_array_equal(extrap10._check_bounds(np.array([-1.0, 0.0, 5.0, 9.0, 11.0])), np.array([[True, False, False, False, False], [False, False, False, False, True]]))
    raises_bounds_error = interp1d(self.x10, self.y10, bounds_error=True, kind=kind)
    self.bounds_check_helper(raises_bounds_error, -1.0, -1.0)
    self.bounds_check_helper(raises_bounds_error, 11.0, 11.0)
    self.bounds_check_helper(raises_bounds_error, [0.0, -1.0, 0.0], -1.0)
    self.bounds_check_helper(raises_bounds_error, [0.0, 1.0, 21.0], 21.0)
    raises_bounds_error([0.0, 5.0, 9.0])