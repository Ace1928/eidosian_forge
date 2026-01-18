import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.tools.tools import add_constant
from statsmodels.base._prediction_inference import PredictionResultsMonotonic
from statsmodels.discrete.discrete_model import (
from statsmodels.discrete.count_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp
class TestZINegativeBinomialPPredict(CheckPredict):

    @classmethod
    def setup_class(cls):
        exog_infl = add_constant(DATA['aget'], prepend=False)
        mod_zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, p=2)
        sp = np.array([-6.58, -1.28, 0.19, 0.08, 0.22, -0.05, 0.03, 0.17, 0.27, 0.68, 0.62])
        res1 = mod_zinb.fit(start_params=sp, method='newton', maxiter=300)
        cls.res1 = res1
        cls.res2 = resp.results_zinb_docvis
        cls.pred_kwds_mean = {'exog_infl': exog_infl.mean(0)}
        cls.pred_kwds_6 = {'exog_infl': exog_infl[:6]}
        cls.k_infl = 2
        cls.rtol = 0.0001