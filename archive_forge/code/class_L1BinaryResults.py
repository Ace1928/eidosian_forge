from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
class L1BinaryResults(BinaryResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'Results instance for binary data fit by l1 regularization', 'extra_attr': _l1_results_attr}

    def __init__(self, model, bnryfit):
        super().__init__(model, bnryfit)
        self.trimmed = bnryfit.mle_retvals['trimmed']
        self.nnz_params = (~self.trimmed).sum()
        self.df_model = self.nnz_params - 1
        self.df_resid = float(self.model.endog.shape[0] - self.nnz_params)