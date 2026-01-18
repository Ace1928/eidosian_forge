import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.special import logit, expit, log_expit
class TestLogExpit:

    def test_large_negative(self):
        x = np.array([-10000.0, -750.0, -500.0, -35.0])
        y = log_expit(x)
        assert_equal(y, x)

    def test_large_positive(self):
        x = np.array([750.0, 1000.0, 10000.0])
        y = log_expit(x)
        assert_equal(y, np.array([-0.0, -0.0, -0.0]))

    def test_basic_float64(self):
        x = np.array([-32, -20, -10, -3, -1, -0.1, -1e-09, 0, 1e-09, 0.1, 1, 10, 100, 500, 710, 725, 735])
        y = log_expit(x)
        expected = [-32.000000000000014, -20.000000002061153, -10.000045398899218, -3.048587351573742, -1.3132616875182228, -0.7443966600735709, -0.6931471810599453, -0.6931471805599453, -0.6931471800599454, -0.6443966600735709, -0.3132616875182228, -4.539889921686465e-05, -3.720075976020836e-44, -7.124576406741286e-218, -4.47628622567513e-309, -1.36930634e-315, -6.217e-320]
        assert_allclose(y, expected, rtol=1e-15)

    def test_basic_float32(self):
        x = np.array([-32, -20, -10, -3, -1, -0.1, -1e-09, 0, 1e-09, 0.1, 1, 10, 100], dtype=np.float32)
        y = log_expit(x)
        expected = np.array([-32.0, -20.0, -10.000046, -3.0485873, -1.3132616, -0.7443967, -0.6931472, -0.6931472, -0.6931472, -0.64439666, -0.3132617, -4.5398898e-05, -3.8e-44], dtype=np.float32)
        assert_allclose(y, expected, rtol=5e-07)