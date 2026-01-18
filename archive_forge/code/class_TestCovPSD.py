import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
class TestCovPSD:

    @classmethod
    def setup_class(cls):
        x = np.array([1, 0.477, 0.644, 0.478, 0.651, 0.826, 0.477, 1, 0.516, 0.233, 0.682, 0.75, 0.644, 0.516, 1, 0.599, 0.581, 0.742, 0.478, 0.233, 0.599, 1, 0.741, 0.8, 0.651, 0.682, 0.581, 0.741, 1, 0.798, 0.826, 0.75, 0.742, 0.8, 0.798, 1]).reshape(6, 6)
        cls.x = x + 0.01 * np.eye(6)
        cls.res = cov_r

    def test_cov_nearest(self):
        x = self.x
        res_r = self.res
        y = cov_nearest(x, method='nearest')
        assert_almost_equal(y, res_r.mat, decimal=3)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.001)
        y = cov_nearest(x, method='clipped')
        assert_almost_equal(y, res_r.mat, decimal=2)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.15)