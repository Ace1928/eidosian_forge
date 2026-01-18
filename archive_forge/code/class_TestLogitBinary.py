import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tools.tools import add_constant
from .results.results_ordinal_model import data_store as ds
class TestLogitBinary:

    def test_attributes(self):
        data = ds.df
        mask_drop = data['apply'] == 'somewhat likely'
        data2 = data.loc[~mask_drop, :].copy()
        data2['apply'] = data2['apply'].cat.remove_categories('somewhat likely')
        modp = OrderedModel(data2['apply'], data2[['pared', 'public', 'gpa']], distr='logit')
        resp = modp.fit(method='bfgs', disp=False)
        exog = add_constant(data2[['pared', 'public', 'gpa']], prepend=False)
        mod_logit = Logit(data2['apply'].cat.codes, exog)
        res_logit = mod_logit.fit()
        attributes = 'bse df_resid llf aic bic llnull'.split()
        attributes += 'llnull llr llr_pvalue prsquared'.split()
        params = np.asarray(resp.params)
        logit_params = np.asarray(res_logit.params)
        assert_allclose(params[:3], logit_params[:3], rtol=1e-05)
        assert_allclose(params[3], -logit_params[3], rtol=1e-05)
        for attr in attributes:
            assert_allclose(getattr(resp, attr), getattr(res_logit, attr), rtol=0.0001)
        resp = modp.fit(method='bfgs', disp=False, cov_type='hac', cov_kwds={'maxlags': 2})
        res_logit = mod_logit.fit(method='bfgs', disp=False, cov_type='hac', cov_kwds={'maxlags': 2})
        for attr in attributes:
            assert_allclose(getattr(resp, attr), getattr(res_logit, attr), rtol=0.0001)
        resp = modp.fit(method='bfgs', disp=False, cov_type='hc1')
        res_logit = mod_logit.fit(method='bfgs', disp=False, cov_type='hc1')
        for attr in attributes:
            assert_allclose(getattr(resp, attr), getattr(res_logit, attr), rtol=0.0001)