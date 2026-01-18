import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
class TestRlmBisquareHuber(TestRlm):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        model = RLM(cls.data.endog, cls.data.exog, M=norms.TukeyBiweight())
        results = model.fit(scale_est=HuberScale())
        h2 = model.fit(cov='H2', scale_est=HuberScale()).bcov_scaled
        h3 = model.fit(cov='H3', scale_est=HuberScale()).bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import BisquareHuber
        self.res2 = BisquareHuber()