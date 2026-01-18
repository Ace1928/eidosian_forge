import numpy as np
import scipy.special as sc
import pytest
from numpy.testing import assert_allclose, assert_array_equal, suppress_warnings
class TestBdtr:

    def test(self):
        val = sc.bdtr(0, 1, 0.5)
        assert_allclose(val, 0.5)

    def test_sum_is_one(self):
        val = sc.bdtr([0, 1, 2], 2, 0.5)
        assert_array_equal(val, [0.25, 0.75, 1.0])

    def test_rounding(self):
        double_val = sc.bdtr([0.1, 1.1, 2.1], 2, 0.5)
        int_val = sc.bdtr([0, 1, 2], 2, 0.5)
        assert_array_equal(double_val, int_val)

    @pytest.mark.parametrize('k, n, p', [(np.inf, 2, 0.5), (1.0, np.inf, 0.5), (1.0, 2, np.inf)])
    def test_inf(self, k, n, p):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            val = sc.bdtr(k, n, p)
        assert np.isnan(val)

    def test_domain(self):
        val = sc.bdtr(-1.1, 1, 0.5)
        assert np.isnan(val)