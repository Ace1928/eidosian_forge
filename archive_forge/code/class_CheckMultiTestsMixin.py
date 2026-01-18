import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
class CheckMultiTestsMixin:

    @pytest.mark.parametrize('key,val', sorted(rmethods.items()))
    def test_multi_pvalcorrection_rmethods(self, key, val):
        res_multtest = self.res2
        pval0 = res_multtest[:, 0]
        if val[1] in self.methods:
            reject, pvalscorr = multipletests(pval0, alpha=self.alpha, method=val[1])[:2]
            assert_almost_equal(pvalscorr, res_multtest[:, val[0]], 15)
            assert_equal(reject, pvalscorr <= self.alpha)

    def test_multi_pvalcorrection(self):
        res_multtest = self.res2
        pval0 = res_multtest[:, 0]
        pvalscorr = np.sort(fdrcorrection(pval0, method='n')[1])
        assert_almost_equal(pvalscorr, res_multtest[:, 7], 15)
        pvalscorr = np.sort(fdrcorrection(pval0, method='i')[1])
        assert_almost_equal(pvalscorr, res_multtest[:, 6], 15)