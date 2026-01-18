import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
def _get_robustcov_results(self, cov_type='opg', **kwargs):
    from statsmodels.base.covtype import descriptions
    use_self = kwargs.pop('use_self', False)
    if use_self:
        res = self
    else:
        raise NotImplementedError
        res = self.__class__(self.model, self.params, normalized_cov_params=self.normalized_cov_params, scale=self.scale)
    res.cov_type = cov_type
    res.cov_kwds = {}
    approx_type_str = 'complex-step'
    k_params = len(self.params)
    if k_params == 0:
        res.cov_params_default = np.zeros((0, 0))
        res._rank = 0
        res.cov_kwds['description'] = 'No parameters estimated.'
    elif cov_type == 'custom':
        res.cov_type = kwargs['custom_cov_type']
        res.cov_params_default = kwargs['custom_cov_params']
        res.cov_kwds['description'] = kwargs['custom_description']
        res._rank = np.linalg.matrix_rank(res.cov_params_default)
    elif cov_type == 'none':
        res.cov_params_default = np.zeros((k_params, k_params)) * np.nan
        res._rank = np.nan
        res.cov_kwds['description'] = descriptions['none']
    elif self.cov_type == 'approx':
        res.cov_params_default = res.cov_params_approx
        res.cov_kwds['description'] = descriptions['approx'].format(approx_type=approx_type_str)
    elif self.cov_type == 'opg':
        res.cov_params_default = res.cov_params_opg
        res.cov_kwds['description'] = descriptions['OPG'].format(approx_type=approx_type_str)
    elif self.cov_type == 'robust':
        res.cov_params_default = res.cov_params_robust
        res.cov_kwds['description'] = descriptions['robust'].format(approx_type=approx_type_str)
    else:
        raise NotImplementedError('Invalid covariance matrix type.')
    return res