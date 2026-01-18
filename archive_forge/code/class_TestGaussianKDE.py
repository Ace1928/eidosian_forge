from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
class TestGaussianKDE:

    def test_kde_integer_input(self):
        """Regression test for #1181."""
        x1 = np.arange(5)
        kde = mlab.GaussianKDE(x1)
        y_expected = [0.13480721, 0.18222869, 0.19514935, 0.18222869, 0.13480721]
        np.testing.assert_array_almost_equal(kde(x1), y_expected, decimal=6)

    def test_gaussian_kde_covariance_caching(self):
        x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
        xs = np.linspace(-10, 10, num=5)
        y_expected = [0.02463386, 0.04689208, 0.05395444, 0.05337754, 0.01664475]
        kde2 = mlab.GaussianKDE(x1, 'scott')
        y2 = kde2(xs)
        np.testing.assert_array_almost_equal(y_expected, y2, decimal=7)

    def test_kde_bandwidth_method(self):
        np.random.seed(8765678)
        n_basesample = 50
        xn = np.random.randn(n_basesample)
        gkde = mlab.GaussianKDE(xn)
        gkde2 = mlab.GaussianKDE(xn, 'scott')
        gkde3 = mlab.GaussianKDE(xn, bw_method=gkde.factor)
        xs = np.linspace(-7, 7, 51)
        kdepdf = gkde.evaluate(xs)
        kdepdf2 = gkde2.evaluate(xs)
        assert kdepdf.all() == kdepdf2.all()
        kdepdf3 = gkde3.evaluate(xs)
        assert kdepdf.all() == kdepdf3.all()