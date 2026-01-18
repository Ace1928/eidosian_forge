import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels import datasets
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.tools.sm_exceptions import (
from statsmodels.distributions.discrete import (
from statsmodels.discrete.truncated_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results.results_discrete import RandHIE
from .results import results_truncated as results_t
from .results import results_truncated_st as results_ts
class TestTruncatedNegBinSt(CheckTruncatedST):

    @classmethod
    def setup_class(cls):
        endog = DATA['docvis']
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedLFNegativeBinomialP(endog, exog).fit(method='bfgs', maxiter=300)
        cls.res2 = results_ts.results_trunc_negbin
        mod_offset = TruncatedLFNegativeBinomialP(endog, exog, offset=DATA['aget'])
        cls.res_offset = mod_offset.fit(method='bfgs', maxiter=300)

    def test_offset(self):
        res1 = self.res1
        reso = self.res_offset
        paramso = np.asarray(reso.params)
        params1 = np.asarray(res1.params)
        assert_allclose(paramso[1:], params1[1:], rtol=1e-08)
        assert_allclose(paramso[0], params1[0] - 1, rtol=1e-08)
        pred1 = res1.predict()
        predo = reso.predict()
        assert_allclose(predo, pred1, rtol=1e-08)
        ex = res1.model.exog[:5]
        offs = reso.model.offset[:5]
        pred1 = res1.predict(ex, transform=False)
        predo = reso.predict(ex, offset=offs, transform=False)
        assert_allclose(predo, pred1, rtol=1e-08)