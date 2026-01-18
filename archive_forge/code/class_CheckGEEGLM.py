import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence
from numpy.testing import assert_allclose
class CheckGEEGLM:

    def test_basic(self):
        res1 = self.result1
        res2 = self.result2
        assert_allclose(res1.params.values, res2.params.values, rtol=1e-06, atol=1e-10)

    def test_resid(self):
        res1 = self.result1
        res2 = self.result2
        assert_allclose(res1.resid_response, res2.resid_response, rtol=1e-06, atol=1e-10)
        assert_allclose(res1.resid_pearson, res2.resid_pearson, rtol=1e-06, atol=1e-10)
        assert_allclose(res1.resid_deviance, res2.resid_deviance, rtol=1e-06, atol=1e-10)
        assert_allclose(res1.resid_anscombe, res2.resid_anscombe, rtol=1e-06, atol=1e-10)
        assert_allclose(res1.resid_working, res2.resid_working, rtol=1e-06, atol=1e-10)