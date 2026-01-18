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
class TruncatedNegativeBinomialResults(TruncatedLFGenericResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Truncated Negative Binomial', 'extra_attr': ''}

    @cache_readonly
    def _dispersion_factor(self):
        if self.model.trunc != 0:
            msg = 'dispersion is only available for zero-truncation'
            raise NotImplementedError(msg)
        alpha = self.params[-1]
        p = self.model.model_main.parameterization
        mu = np.exp(self.predict(which='linear'))
        return 1 - alpha * mu ** (p - 1) / (np.exp(mu ** (p - 1)) - 1)