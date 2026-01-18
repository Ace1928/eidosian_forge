import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
from statsmodels.distributions import zipoisson, zigenpoisson, zinegbin
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.compat.pandas import Appender
class ZeroInflatedNegativeBinomialResults(ZeroInflatedResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Zero Inflated Generalized Negative Binomial', 'extra_attr': ''}

    @cache_readonly
    def _dispersion_factor(self):
        p = self.model.model_main.parameterization
        alpha = self.params[self.model.k_inflate:][-1]
        mu = np.exp(self.predict(which='linear'))
        w = 1 - self.predict() / mu
        return 1 + alpha * mu ** (p - 1) + w * mu

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        raise NotImplementedError('not yet implemented for zero inflation')