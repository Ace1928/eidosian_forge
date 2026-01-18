import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_equal
from statsmodels.regression.linear_model import GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
class CheckStataResultsMixin:

    def test_params_table(self):
        res, results = (self.res, self.results)
        assert_almost_equal(res.params, results.params, 3)
        assert_almost_equal(res.bse, results.bse, 3)
        assert_allclose(res.tvalues, results.tvalues, atol=0, rtol=0.004)
        assert_allclose(res.pvalues, results.pvalues, atol=1e-07, rtol=0.004)