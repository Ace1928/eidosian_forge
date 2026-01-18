import numpy as np
from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere
from statsmodels.base.model import GenericLikelihoodModel
class zigeneralizedpoisson_gen(rv_discrete):
    """Zero Inflated Generalized Poisson distribution
    """

    def _argcheck(self, mu, alpha, p, w):
        return (mu > 0) & (w >= 0) & (w <= 1)

    def _logpmf(self, x, mu, alpha, p, w):
        return _lazywhere(x != 0, (x, mu, alpha, p, w), lambda x, mu, alpha, p, w: np.log(1.0 - w) + genpoisson_p.logpmf(x, mu, alpha, p), np.log(w + (1.0 - w) * genpoisson_p.pmf(x, mu, alpha, p)))

    def _pmf(self, x, mu, alpha, p, w):
        return np.exp(self._logpmf(x, mu, alpha, p, w))

    def mean(self, mu, alpha, p, w):
        return (1 - w) * mu

    def var(self, mu, alpha, p, w):
        p = p - 1
        dispersion_factor = (1 + alpha * mu ** p) ** 2 + w * mu
        var = dispersion_factor * self.mean(mu, alpha, p, w)
        return var