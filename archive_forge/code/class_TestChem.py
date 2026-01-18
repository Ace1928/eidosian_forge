import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
class TestChem:

    @classmethod
    def setup_class(cls):
        cls.chem = np.array([2.2, 2.2, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03, 3.03, 3.1, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7, 3.7, 3.7, 3.77, 5.28, 28.95])

    def test_mean(self):
        assert_almost_equal(np.mean(self.chem), 4.2804, DECIMAL)

    def test_median(self):
        assert_almost_equal(np.median(self.chem), 3.385, DECIMAL)

    def test_mad(self):
        assert_almost_equal(scale.mad(self.chem), 0.52632, DECIMAL)

    def test_iqr(self):
        assert_almost_equal(scale.iqr(self.chem), 0.6857, DECIMAL)

    def test_qn(self):
        assert_almost_equal(scale.qn_scale(self.chem), 0.73231, DECIMAL)

    def test_huber_scale(self):
        assert_almost_equal(scale.huber(self.chem)[0], 3.20549, DECIMAL)

    def test_huber_location(self):
        assert_almost_equal(scale.huber(self.chem)[1], 0.67365, DECIMAL)

    def test_huber_huberT(self):
        n = scale.norms.HuberT()
        n.t = 1.5
        h = scale.Huber(norm=n)
        assert_almost_equal(scale.huber(self.chem)[0], h(self.chem)[0], DECIMAL)
        assert_almost_equal(scale.huber(self.chem)[1], h(self.chem)[1], DECIMAL)

    def test_huber_Hampel(self):
        hh = scale.Huber(norm=scale.norms.Hampel())
        assert_almost_equal(hh(self.chem)[0], 3.17434, DECIMAL)
        assert_almost_equal(hh(self.chem)[1], 0.66782, DECIMAL)