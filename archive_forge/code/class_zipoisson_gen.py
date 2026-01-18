import numpy as np
from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere
from statsmodels.base.model import GenericLikelihoodModel
class zipoisson_gen(rv_discrete):
    """Zero Inflated Poisson distribution
    """

    def _argcheck(self, mu, w):
        return (mu > 0) & (w >= 0) & (w <= 1)

    def _logpmf(self, x, mu, w):
        return _lazywhere(x != 0, (x, mu, w), lambda x, mu, w: np.log(1.0 - w) + x * np.log(mu) - gammaln(x + 1.0) - mu, np.log(w + (1.0 - w) * np.exp(-mu)))

    def _pmf(self, x, mu, w):
        return np.exp(self._logpmf(x, mu, w))

    def _cdf(self, x, mu, w):
        return w + poisson(mu=mu).cdf(x) * (1 - w)

    def _ppf(self, q, mu, w):
        q_mod = (q - w) / (1 - w)
        x = poisson(mu=mu).ppf(q_mod)
        x[q < w] = 0
        return x

    def mean(self, mu, w):
        return (1 - w) * mu

    def var(self, mu, w):
        dispersion_factor = 1 + w * mu
        var = dispersion_factor * self.mean(mu, w)
        return var

    def _moment(self, n, mu, w):
        return (1 - w) * poisson.moment(n, mu)