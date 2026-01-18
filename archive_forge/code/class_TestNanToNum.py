import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestNanToNum:

    def test_generic(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0)
        assert_all(vals[0] < -10000000000.0) and assert_all(np.isfinite(vals[0]))
        assert_(vals[1] == 0)
        assert_all(vals[2] > 10000000000.0) and assert_all(np.isfinite(vals[2]))
        assert_equal(type(vals), np.ndarray)
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=10, posinf=20, neginf=30)
        assert_equal(vals, [30, 10, 20])
        assert_all(np.isfinite(vals[[0, 2]]))
        assert_equal(type(vals), np.ndarray)
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = np.array((-1.0, 0, 1)) / 0.0
        result = nan_to_num(vals, copy=False)
        assert_(result is vals)
        assert_all(vals[0] < -10000000000.0) and assert_all(np.isfinite(vals[0]))
        assert_(vals[1] == 0)
        assert_all(vals[2] > 10000000000.0) and assert_all(np.isfinite(vals[2]))
        assert_equal(type(vals), np.ndarray)
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = np.array((-1.0, 0, 1)) / 0.0
        result = nan_to_num(vals, copy=False, nan=10, posinf=20, neginf=30)
        assert_(result is vals)
        assert_equal(vals, [30, 10, 20])
        assert_all(np.isfinite(vals[[0, 2]]))
        assert_equal(type(vals), np.ndarray)

    def test_array(self):
        vals = nan_to_num([1])
        assert_array_equal(vals, np.array([1], int))
        assert_equal(type(vals), np.ndarray)
        vals = nan_to_num([1], nan=10, posinf=20, neginf=30)
        assert_array_equal(vals, np.array([1], int))
        assert_equal(type(vals), np.ndarray)

    def test_integer(self):
        vals = nan_to_num(1)
        assert_all(vals == 1)
        assert_equal(type(vals), np.int_)
        vals = nan_to_num(1, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1)
        assert_equal(type(vals), np.int_)

    def test_float(self):
        vals = nan_to_num(1.0)
        assert_all(vals == 1.0)
        assert_equal(type(vals), np.float_)
        vals = nan_to_num(1.1, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1.1)
        assert_equal(type(vals), np.float_)

    def test_complex_good(self):
        vals = nan_to_num(1 + 1j)
        assert_all(vals == 1 + 1j)
        assert_equal(type(vals), np.complex_)
        vals = nan_to_num(1 + 1j, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1 + 1j)
        assert_equal(type(vals), np.complex_)

    def test_complex_bad(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            v = 1 + 1j
            v += np.array(0 + 1j) / 0.0
        vals = nan_to_num(v)
        assert_all(np.isfinite(vals))
        assert_equal(type(vals), np.complex_)

    def test_complex_bad2(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            v = 1 + 1j
            v += np.array(-1 + 1j) / 0.0
        vals = nan_to_num(v)
        assert_all(np.isfinite(vals))
        assert_equal(type(vals), np.complex_)

    def test_do_not_rewrite_previous_keyword(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=np.inf, posinf=999)
        assert_all(np.isfinite(vals[[0, 2]]))
        assert_all(vals[0] < -10000000000.0)
        assert_equal(vals[[1, 2]], [np.inf, 999])
        assert_equal(type(vals), np.ndarray)