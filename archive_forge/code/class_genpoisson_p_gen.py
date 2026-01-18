import numpy as np
from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere
from statsmodels.base.model import GenericLikelihoodModel
class genpoisson_p_gen(rv_discrete):
    """Generalized Poisson distribution
    """

    def _argcheck(self, mu, alpha, p):
        return (mu >= 0) & (alpha == alpha) & (p > 0)

    def _logpmf(self, x, mu, alpha, p):
        mu_p = mu ** (p - 1.0)
        a1 = np.maximum(np.nextafter(0, 1), 1 + alpha * mu_p)
        a2 = np.maximum(np.nextafter(0, 1), mu + (a1 - 1.0) * x)
        logpmf_ = np.log(mu) + (x - 1.0) * np.log(a2)
        logpmf_ -= x * np.log(a1) + gammaln(x + 1.0) + a2 / a1
        return logpmf_

    def _pmf(self, x, mu, alpha, p):
        return np.exp(self._logpmf(x, mu, alpha, p))

    def mean(self, mu, alpha, p):
        return mu

    def var(self, mu, alpha, p):
        dispersion_factor = (1 + alpha * mu ** (p - 1)) ** 2
        var = dispersion_factor * mu
        return var