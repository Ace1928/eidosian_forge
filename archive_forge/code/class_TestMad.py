import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
class TestMad:

    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10))

    def test_mad(self):
        m = scale.mad(self.X)
        assert_equal(m.shape, (10,))

    def test_mad_empty(self):
        empty = np.empty(0)
        assert np.isnan(scale.mad(empty))
        empty = np.empty((10, 100, 0))
        assert_equal(scale.mad(empty, axis=1), np.empty((10, 0)))
        empty = np.empty((100, 100, 0, 0))
        assert_equal(scale.mad(empty, axis=-1), np.empty((100, 100, 0)))

    def test_mad_center(self):
        n = scale.mad(self.X, center=0)
        assert_equal(n.shape, (10,))
        with pytest.raises(TypeError):
            scale.mad(self.X, center=None)
        assert_almost_equal(scale.mad(self.X, center=1), np.median(np.abs(self.X - 1), axis=0) / Gaussian.ppf(3 / 4.0), DECIMAL)