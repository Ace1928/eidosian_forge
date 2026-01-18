import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
class TestCpow:

    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)

    def test_simple(self):
        x = np.array([1 + 1j, 0 + 2j, 1 + 2j, np.inf, np.nan])
        y_r = x ** 2
        y = np.power(x, 2)
        assert_almost_equal(y, y_r)

    def test_scalar(self):
        x = np.array([1, 1j, 2, 2.5 + 0.37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5 + 1.5j, -0.5 + 1.5j, 2, 3])
        lx = list(range(len(x)))
        p_r = [1 + 0j, 0.20787957635076193 + 0j, 0.35812203996480685 + 0.6097119028618724j, 0.12659112128185032 + 0.48847676699581527j, complex(np.inf, np.nan), complex(np.nan, np.nan)]
        n_r = [x[i] ** y[i] for i in lx]
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)

    def test_array(self):
        x = np.array([1, 1j, 2, 2.5 + 0.37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5 + 1.5j, -0.5 + 1.5j, 2, 3])
        lx = list(range(len(x)))
        p_r = [1 + 0j, 0.20787957635076193 + 0j, 0.35812203996480685 + 0.6097119028618724j, 0.12659112128185032 + 0.48847676699581527j, complex(np.inf, np.nan), complex(np.nan, np.nan)]
        n_r = x ** y
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)