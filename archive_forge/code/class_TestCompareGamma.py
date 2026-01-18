import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence
from numpy.testing import assert_allclose
class TestCompareGamma(CheckGEEGLM):

    @classmethod
    def setup_class(cls):
        vs = Independence()
        family = families.Gamma(link=links.Log())
        np.random.seed(987126)
        Y = np.exp(0.1 + np.random.normal(size=100))
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)
        D = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3})
        mod1 = GEE.from_formula('Y ~ X1 + X2 + X3', groups, D, family=family, cov_struct=vs)
        cls.result1 = mod1.fit()
        mod2 = GLM.from_formula('Y ~ X1 + X2 + X3', data=D, family=family)
        cls.result2 = mod2.fit(disp=False)