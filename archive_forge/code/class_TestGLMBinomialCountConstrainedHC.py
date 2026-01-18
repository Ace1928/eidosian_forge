import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
class TestGLMBinomialCountConstrainedHC(TestGLMBinomialCountConstrained):

    @classmethod
    def init(cls):
        cls.res2 = cls.mod2.fit(cov_type='HC0')
        k = cls.mod1.exog.shape[1]
        cls.idx_p_uc = np.arange(k - 5)
        constraints = np.eye(k)[-5:]
        cls.res1 = cls.mod1.fit_constrained(constraints, cov_type='HC0')