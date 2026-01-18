import pytest
import numpy as np
from scipy import stats
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.sandbox.distributions.extras import NormExpan_gen
class CheckDistribution:

    @pytest.mark.smoke
    def test_dist1(self):
        self.dist1.rvs(size=10)
        self.dist1.pdf(np.linspace(-4, 4, 11))

    def test_cdf_ppf_roundtrip(self):
        probs = np.linspace(0.001, 0.999, 6)
        ppf = self.dist2.ppf(probs)
        cdf = self.dist2.cdf(ppf)
        assert_allclose(cdf, probs, rtol=1e-06)
        sf = self.dist2.sf(ppf)
        assert_allclose(sf, 1 - probs, rtol=1e-06)