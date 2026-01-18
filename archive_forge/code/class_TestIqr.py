import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
class TestIqr:

    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10))

    def test_iqr(self):
        m = scale.iqr(self.X)
        assert_equal(m.shape, (10,))

    def test_iqr_empty(self):
        empty = np.empty(0)
        assert np.isnan(scale.iqr(empty))
        empty = np.empty((10, 100, 0))
        assert_equal(scale.iqr(empty, axis=1), np.empty((10, 0)))
        empty = np.empty((100, 100, 0, 0))
        assert_equal(scale.iqr(empty, axis=-1), np.empty((100, 100, 0)))
        empty = np.empty(shape=())
        with pytest.raises(ValueError):
            scale.iqr(empty)