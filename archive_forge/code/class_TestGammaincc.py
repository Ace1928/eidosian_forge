import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.special as sc
from scipy.special._testutils import FuncData
class TestGammaincc:

    @pytest.mark.parametrize('a, x', INVALID_POINTS)
    def test_domain(self, a, x):
        assert np.isnan(sc.gammaincc(a, x))

    def test_a_eq_0_x_gt_0(self):
        assert sc.gammaincc(0, 1) == 0

    @pytest.mark.parametrize('a, x, desired', [(np.inf, 1, 1), (np.inf, 0, 1), (np.inf, np.inf, np.nan), (1, np.inf, 0)])
    def test_infinite_arguments(self, a, x, desired):
        result = sc.gammaincc(a, x)
        if np.isnan(desired):
            assert np.isnan(result)
        else:
            assert result == desired

    def test_infinite_limits(self):
        assert sc.gammaincc(1000, 100) == sc.gammaincc(np.inf, 100)
        assert_allclose(sc.gammaincc(100, 1000), sc.gammaincc(100, np.inf), atol=1e-200, rtol=0)

    def test_limit_check(self):
        result = sc.gammaincc(1e-10, 1)
        limit = sc.gammaincc(0, 1)
        assert np.isclose(result, limit)

    def test_x_zero(self):
        a = np.arange(1, 10)
        assert_array_equal(sc.gammaincc(a, 0), 1)

    def test_roundtrip(self):
        a = np.logspace(-5, 10, 100)
        x = np.logspace(-5, 10, 100)
        y = sc.gammainccinv(a, sc.gammaincc(a, x))
        assert_allclose(x, y, rtol=1e-14)