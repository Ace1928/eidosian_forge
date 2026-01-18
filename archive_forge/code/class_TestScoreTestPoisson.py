import numpy as np
from numpy.testing import assert_allclose
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Poisson
import statsmodels.stats._diagnostic_other as diao
import statsmodels.discrete._diagnostics_count as diac
from statsmodels.base._parameter_inference import score_test
class TestScoreTestPoisson(TestScoreTest):
    rtol_ws = 0.005
    atol_ws = 0
    rtol_wooldridge = 0.004
    dispersed = False
    res_pvalue = [0.31786373532550893, 0.32654081685271297]
    skip_wooldridge = False
    res_disptest = np.array([[0.1392791916012637, 0.8892295323009857], [0.1392791916012645, 0.889229532300985], [0.2129554490802097, 0.8313617120611572], [0.1493501809372359, 0.881277320588635], [0.1493501809372359, 0.881277320588635], [0.1454862255574059, 0.8843269904545624], [0.2281321688124869, 0.8195434922982738]])
    res_disptest_g = [0.05224762959371576, 0.8191973886772222]

    @classmethod
    def setup_class(cls):
        nobs, k_vars = (500, 5)
        np.random.seed(786452)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1
        x2 = np.random.randn(nobs, 2)
        xx = np.column_stack((x, x2))
        if cls.dispersed:
            het = np.random.randn(nobs)
            y = np.random.poisson(np.exp(x.sum(1) * 0.5 + het))
        else:
            y = np.random.poisson(np.exp(x.sum(1) * 0.5))
        cls.exog_extra = x2
        cls.model_full = Poisson(y, xx)
        cls.model_drop = Poisson(y, x)

    def test_wald_score(self):
        super().test_wald_score()