import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pytest
from statsmodels.stats.robust_compare import (
import statsmodels.stats.oneway as smo
from statsmodels.tools.testing import Holder
from scipy.stats import trim1
class Test_Trim:

    def t_est_trim1(self):
        a = np.arange(11)
        assert_equal(trim1(a, 0.1), np.arange(10))
        assert_equal(trim1(a, 0.2), np.arange(9))
        assert_equal(trim1(a, 0.2, tail='left'), np.arange(2, 11))
        assert_equal(trim1(a, 3 / 11.0, tail='left'), np.arange(3, 11))

    def test_trimboth(self):
        a = np.arange(11)
        a2 = np.arange(24).reshape(6, 4)
        a3 = np.arange(24).reshape(6, 4, order='F')
        assert_equal(trimboth(a, 3 / 11.0), np.arange(3, 8))
        assert_equal(trimboth(a, 0.2), np.array([2, 3, 4, 5, 6, 7, 8]))
        assert_equal(trimboth(a2, 0.2), np.arange(4, 20).reshape(4, 4))
        assert_equal(trimboth(a3, 2 / 6.0), np.array([[2, 8, 14, 20], [3, 9, 15, 21]]))
        assert_raises(ValueError, trimboth, np.arange(24).reshape(4, 6).T, 4 / 6.0)

    def test_trim_mean(self):
        idx = np.array([3, 5, 0, 1, 2, 4])
        a2 = np.arange(24).reshape(6, 4)[idx, :]
        a3 = np.arange(24).reshape(6, 4, order='F')[idx, :]
        assert_equal(trim_mean(a3, 2 / 6.0), np.array([2.5, 8.5, 14.5, 20.5]))
        assert_equal(trim_mean(a2, 2 / 6.0), np.array([10.0, 11.0, 12.0, 13.0]))
        idx4 = np.array([1, 0, 3, 2])
        a4 = np.arange(24).reshape(4, 6)[idx4, :]
        assert_equal(trim_mean(a4, 2 / 6.0), np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0]))
        a = np.array([7, 11, 12, 21, 16, 6, 22, 1, 5, 0, 18, 10, 17, 9, 19, 15, 23, 20, 2, 14, 4, 13, 8, 3])
        assert_equal(trim_mean(a, 2 / 6.0), 11.5)
        assert_equal(trim_mean([5, 4, 3, 1, 2, 0], 2 / 6.0), 2.5)
        np.random.seed(1234)
        a = np.random.randint(20, size=(5, 6, 4, 7))
        for axis in [0, 1, 2, 3, -1]:
            res1 = trim_mean(a, 2 / 6.0, axis=axis)
            res2 = trim_mean(np.rollaxis(a, axis), 2 / 6.0)
            assert_equal(res1, res2)
        res1 = trim_mean(a, 2 / 6.0, axis=None)
        res2 = trim_mean(a.ravel(), 2 / 6.0)
        assert_equal(res1, res2)