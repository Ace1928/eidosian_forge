import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
class TestNdtri:

    def test_zero(self):
        assert sc.ndtri(0.5) == 0.0

    def test_asymptotes(self):
        assert_equal(sc.ndtri([0.0, 1.0]), [-np.inf, np.inf])

    def test_outside_of_domain(self):
        assert all(np.isnan(sc.ndtri([-1.5, 1.5])))