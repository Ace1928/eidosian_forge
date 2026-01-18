import numpy as np
from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere
from statsmodels.base.model import GenericLikelihoodModel
class truncatedpoisson_gen(rv_discrete):
    """Truncated Poisson discrete random variable
    """

    def _argcheck(self, mu, truncation):
        return (mu >= 0) & (truncation >= -1)

    def _get_support(self, mu, truncation):
        return (truncation + 1, self.b)

    def _logpmf(self, x, mu, truncation):
        pmf = 0
        for i in range(int(np.max(truncation)) + 1):
            pmf += poisson.pmf(i, mu)
        log_1_m_pmf = np.full_like(pmf, -np.inf)
        loc = pmf > 1
        log_1_m_pmf[loc] = np.nan
        loc = pmf < 1
        log_1_m_pmf[loc] = np.log(1 - pmf[loc])
        logpmf_ = poisson.logpmf(x, mu) - log_1_m_pmf
        return logpmf_

    def _pmf(self, x, mu, truncation):
        return np.exp(self._logpmf(x, mu, truncation))