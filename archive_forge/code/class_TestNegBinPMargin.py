import numpy as np
from numpy.testing import assert_allclose
from statsmodels.datasets.cpunish import load
from statsmodels.discrete.discrete_model import (
import statsmodels.discrete.tests.results.results_count_margins as res_stata
from statsmodels.tools.tools import add_constant
class TestNegBinPMargin(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        start_params = [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552, -2.88, 1.14]
        mod = NegativeBinomialP(endog, exog)
        res = mod.fit(start_params=start_params, method='nm', maxiter=2000)
        marge = res.get_margeff()
        cls.res = res
        cls.margeff = marge
        cls.res1_slice = slice(None, None, None)
        cls.res1 = res_stata.results_negbin_margins_cont
        cls.rtol_fac = 50.0