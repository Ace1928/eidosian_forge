from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def _cov_params(self, **kwds):
    if 'wargs' not in kwds:
        kwds['wargs'] = self.wargs
    if 'weights_method' not in kwds:
        kwds['weights_method'] = self.options_other['weights_method']
    if 'has_optimal_weights' not in kwds:
        kwds['has_optimal_weights'] = self.options_other['has_optimal_weights']
    gradmoms = self.model.gradient_momcond(self.params)
    moms = self.model.momcond(self.params)
    covparams = self.calc_cov_params(moms, gradmoms, **kwds)
    return covparams