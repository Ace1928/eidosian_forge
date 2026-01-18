import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
class TestGLMGaussianOffsetHC(ConstrainedCompareMixin):

    @classmethod
    def init(cls):
        cov_type = 'HC0'
        cls.res2 = cls.mod2.fit(cov_type=cov_type)
        mod = GLM(cls.endog, cls.exogc, offset=0.5 * cls.exog[:, cls.idx_c].squeeze())
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit(cov_type=cov_type)
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])