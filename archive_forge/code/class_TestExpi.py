import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
class TestExpi:

    @pytest.mark.parametrize('result', [sc.expi(complex(-1, 0)), sc.expi(complex(-1, -0.0)), sc.expi(-1)])
    def test_branch_cut(self, result):
        desired = -0.21938393439552029
        assert_allclose(result, desired, atol=0, rtol=1e-14)

    def test_near_branch_cut(self):
        lim_from_above = sc.expi(-1 + 1e-20j)
        lim_from_below = sc.expi(-1 - 1e-20j)
        assert_allclose(lim_from_above.real, lim_from_below.real, atol=0, rtol=1e-15)
        assert_allclose(lim_from_above.imag, -lim_from_below.imag, atol=0, rtol=1e-15)

    def test_continuity_on_positive_real_axis(self):
        assert_allclose(sc.expi(complex(1, 0)), sc.expi(complex(1, -0.0)), atol=0, rtol=1e-15)