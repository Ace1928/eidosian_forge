import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
import numpy.testing as npt
from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats
from statsmodels.distributions.edgeworth import (_faa_di_bruno_partitions,
class TestCumulants:

    def test_badvalues(self):
        assert_raises(ValueError, cumulant_from_moments, [1, 2, 3], 0)
        assert_raises(ValueError, cumulant_from_moments, [1, 2, 3], 4)

    def test_norm(self):
        N = 4
        momt = [_norm_moment(j + 1) for j in range(N)]
        for n in range(1, N + 1):
            kappa = cumulant_from_moments(momt, n)
            assert_allclose(kappa, _norm_cumulant(n), atol=1e-12)

    def test_chi2(self):
        N = 4
        df = 8
        momt = [_chi2_moment(j + 1, df) for j in range(N)]
        for n in range(1, N + 1):
            kappa = cumulant_from_moments(momt, n)
            assert_allclose(kappa, _chi2_cumulant(n, df))