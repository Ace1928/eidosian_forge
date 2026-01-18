import pytest
import numpy as np
from scipy import stats
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.sandbox.distributions.extras import NormExpan_gen
class CheckExpandNorm(CheckDistribution):

    def test_pdf(self):
        scale = getattr(self, 'scale', 1)
        x = np.linspace(-4, 4, 11) * scale
        pdf2 = self.dist2.pdf(x)
        pdf1 = self.dist1.pdf(x)
        atol_pdf = getattr(self, 'atol_pdf', 0)
        assert_allclose(((pdf2 - pdf1) ** 2).mean(), 0, rtol=1e-06, atol=atol_pdf)
        assert_allclose(pdf2, pdf1, rtol=1e-06, atol=atol_pdf)

    def test_mvsk(self):
        mvsk2 = self.dist2.mvsk
        mvsk1 = self.dist2.stats(moments='mvsk')
        assert_allclose(mvsk2, mvsk1, rtol=1e-06, atol=1e-13)
        assert_allclose(self.dist2.mvsk, self.mvsk, rtol=1e-12)