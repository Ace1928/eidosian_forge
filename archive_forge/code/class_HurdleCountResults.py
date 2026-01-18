import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.distributions.discrete import (
from statsmodels.discrete.discrete_model import (
from statsmodels.tools.numdiff import approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from copy import deepcopy
class HurdleCountResults(CountResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Hurdle model', 'extra_attr': ''}

    def __init__(self, model, mlefit, results_zero, results_count, cov_type='nonrobust', cov_kwds=None, use_t=None):
        super().__init__(model, mlefit, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        self.results_zero = results_zero
        self.results_count = results_count
        self.df_resid = self.model.endog.shape[0] - len(self.params)

    @cache_readonly
    def llnull(self):
        return self.results_zero._results.llnull + self.results_count._results.llnull

    @cache_readonly
    def bse(self):
        return np.append(self.results_zero.bse, self.results_count.bse)