import numpy as np
from numpy.testing import assert_allclose
from statsmodels.datasets.cpunish import load
from statsmodels.discrete.discrete_model import (
import statsmodels.discrete.tests.results.results_count_margins as res_stata
from statsmodels.tools.tools import add_constant
class TestPoissonMargin(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        start_params = [14.1709, 0.7085, -3.4548, -0.539, 3.2368, -7.9299, -5.0529]
        mod_poi = Poisson(endog, exog)
        res_poi = mod_poi.fit(start_params=start_params)
        marge_poi = res_poi.get_margeff()
        cls.res = res_poi
        cls.margeff = marge_poi
        cls.rtol_fac = 1
        cls.res1_slice = slice(None, None, None)
        cls.res1 = res_stata.results_poisson_margins_cont