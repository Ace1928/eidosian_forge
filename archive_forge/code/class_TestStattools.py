import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
class TestStattools:

    @classmethod
    def setup_class(cls):
        x = np.random.standard_normal(1000)
        e1, e2, e3, e4, e5, e6, e7 = np.percentile(x, (12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5))
        c05, c50, c95 = np.percentile(x, (5.0, 50.0, 95.0))
        f025, f25, f75, f975 = np.percentile(x, (2.5, 25.0, 75.0, 97.5))
        mean = np.mean
        kr1 = mean(((x - mean(x)) / np.std(x)) ** 4.0) - 3.0
        kr2 = (e7 - e5 + (e3 - e1)) / (e6 - e2) - 1.2330951154852172
        kr3 = (mean(x[x > c95]) - mean(x[x < c05])) / (mean(x[x > c50]) - mean(x[x < c50])) - 2.585227122870805
        kr4 = (f975 - f025) / (f75 - f25) - 2.905846951670164
        cls.kurtosis_x = x
        cls.expected_kurtosis = np.array([kr1, kr2, kr3, kr4])
        cls.kurtosis_constants = np.array([3.0, 1.2330951154852172, 2.585227122870805, 2.905846951670164])

    def test_medcouple_no_axis(self):
        x = np.reshape(np.arange(100.0), (50, 2))
        mc = medcouple(x, axis=None)
        assert_almost_equal(mc, medcouple(x.ravel()))

    def test_medcouple_1d(self):
        x = np.reshape(np.arange(100.0), (50, 2))
        assert_raises(ValueError, _medcouple_1d, x)

    def test_medcouple_symmetric(self):
        mc = medcouple(np.arange(5.0))
        assert_almost_equal(mc, 0)

    def test_medcouple_nonzero(self):
        mc = medcouple(np.array([1, 2, 7, 9, 10.0]))
        assert_almost_equal(mc, -0.3333333)

    def test_medcouple_int(self):
        mc1 = medcouple(np.array([1, 2, 7, 9, 10]))
        mc2 = medcouple(np.array([1, 2, 7, 9, 10.0]))
        assert_equal(mc1, mc2)

    def test_medcouple_symmetry(self, reset_randomstate):
        x = np.random.standard_normal(100)
        mcp = medcouple(x)
        mcn = medcouple(-x)
        assert_almost_equal(mcp + mcn, 0)

    def test_medcouple_ties(self, reset_randomstate):
        x = np.array([1, 2, 2, 3, 4])
        mc = medcouple(x)
        assert_almost_equal(mc, 1.0 / 6.0)

    def test_durbin_watson(self, reset_randomstate):
        x = np.random.standard_normal(100)
        dw = sum(np.diff(x) ** 2.0) / np.dot(x, x)
        assert_almost_equal(dw, durbin_watson(x))

    def test_durbin_watson_2d(self, reset_randomstate):
        shape = (1, 10)
        x = np.random.standard_normal(100)
        dw = sum(np.diff(x) ** 2.0) / np.dot(x, x)
        x = np.tile(x[:, None], shape)
        assert_almost_equal(np.squeeze(dw * np.ones(shape)), durbin_watson(x))

    def test_durbin_watson_3d(self, reset_randomstate):
        shape = (10, 1, 10)
        x = np.random.standard_normal(100)
        dw = sum(np.diff(x) ** 2.0) / np.dot(x, x)
        x = np.tile(x[None, :, None], shape)
        assert_almost_equal(np.squeeze(dw * np.ones(shape)), durbin_watson(x, axis=1))

    def test_robust_skewness_1d(self):
        x = np.arange(21.0)
        sk = robust_skewness(x)
        assert_almost_equal(np.array(sk), np.zeros(4))

    def test_robust_skewness_1d_2d(self, reset_randomstate):
        x = np.random.randn(21)
        y = x[:, None]
        sk_x = robust_skewness(x)
        sk_y = robust_skewness(y, axis=None)
        assert_almost_equal(np.array(sk_x), np.array(sk_y))

    def test_robust_skewness_symmetric(self, reset_randomstate):
        x = np.random.standard_normal(100)
        x = np.hstack([x, np.zeros(1), -x])
        sk = robust_skewness(x)
        assert_almost_equal(np.array(sk), np.zeros(4))

    def test_robust_skewness_3d(self, reset_randomstate):
        x = np.random.standard_normal(100)
        x = np.hstack([x, np.zeros(1), -x])
        x = np.tile(x, (10, 10, 1))
        sk_3d = robust_skewness(x, axis=2)
        result = np.zeros((10, 10))
        for sk in sk_3d:
            assert_almost_equal(sk, result)

    def test_robust_skewness_4(self, reset_randomstate):
        x = np.random.standard_normal(1000)
        x[x > 0] *= 3
        m = np.median(x)
        s = x.std(ddof=0)
        expected = (x.mean() - m) / s
        _, _, _, sk4 = robust_skewness(x)
        assert_allclose(expected, sk4)

    def test_robust_kurtosis_1d_2d(self, reset_randomstate):
        x = np.random.randn(100)
        y = x[:, None]
        kr_x = np.array(robust_kurtosis(x))
        kr_y = np.array(robust_kurtosis(y, axis=None))
        assert_almost_equal(kr_x, kr_y)

    def test_robust_kurtosis(self):
        x = self.kurtosis_x
        assert_almost_equal(np.array(robust_kurtosis(x)), self.expected_kurtosis)

    def test_robust_kurtosis_3d(self):
        x = np.tile(self.kurtosis_x, (10, 10, 1))
        kurtosis = np.array(robust_kurtosis(x, axis=2))
        for i, r in enumerate(self.expected_kurtosis):
            assert_almost_equal(r * np.ones((10, 10)), kurtosis[i])

    def test_robust_kurtosis_excess_false(self):
        x = self.kurtosis_x
        expected = self.expected_kurtosis + self.kurtosis_constants
        kurtosis = np.array(robust_kurtosis(x, excess=False))
        assert_almost_equal(expected, kurtosis)

    def test_robust_kurtosis_ab(self):
        x = self.kurtosis_x
        alpha, beta = (10.0, 45.0)
        kurtosis = robust_kurtosis(self.kurtosis_x, ab=(alpha, beta), excess=False)
        num = np.mean(x[x > np.percentile(x, 100.0 - alpha)]) - np.mean(x[x < np.percentile(x, alpha)])
        denom = np.mean(x[x > np.percentile(x, 100.0 - beta)]) - np.mean(x[x < np.percentile(x, beta)])
        assert_almost_equal(kurtosis[2], num / denom)

    def test_robust_kurtosis_dg(self):
        x = self.kurtosis_x
        delta, gamma = (10.0, 45.0)
        kurtosis = robust_kurtosis(self.kurtosis_x, dg=(delta, gamma), excess=False)
        q = np.percentile(x, [delta, 100.0 - delta, gamma, 100.0 - gamma])
        assert_almost_equal(kurtosis[3], (q[1] - q[0]) / (q[3] - q[2]))