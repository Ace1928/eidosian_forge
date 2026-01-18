import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
from statsmodels.tools.testing import Holder
class CheckCohens:

    def test_results(self):
        res = self.res
        res2 = self.res2
        res_ = [res.kappa, res.std_kappa, res.kappa_low, res.kappa_upp, res.std_kappa0, res.z_value, res.pvalue_one_sided, res.pvalue_two_sided]
        assert_almost_equal(res_, res2, decimal=4)
        assert_equal(str(res), self.res_string)