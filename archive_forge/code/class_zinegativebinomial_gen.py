import numpy as np
from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere
from statsmodels.base.model import GenericLikelihoodModel
class zinegativebinomial_gen(rv_discrete):
    """Zero Inflated Generalized Negative Binomial distribution
    """

    def _argcheck(self, mu, alpha, p, w):
        return (mu > 0) & (w >= 0) & (w <= 1)

    def _logpmf(self, x, mu, alpha, p, w):
        s, p = self.convert_params(mu, alpha, p)
        return _lazywhere(x != 0, (x, s, p, w), lambda x, s, p, w: np.log(1.0 - w) + nbinom.logpmf(x, s, p), np.log(w + (1.0 - w) * nbinom.pmf(x, s, p)))

    def _pmf(self, x, mu, alpha, p, w):
        return np.exp(self._logpmf(x, mu, alpha, p, w))

    def _cdf(self, x, mu, alpha, p, w):
        s, p = self.convert_params(mu, alpha, p)
        return w + nbinom.cdf(x, s, p) * (1 - w)

    def _ppf(self, q, mu, alpha, p, w):
        s, p = self.convert_params(mu, alpha, p)
        q_mod = (q - w) / (1 - w)
        x = nbinom.ppf(q_mod, s, p)
        x[q < w] = 0
        return x

    def mean(self, mu, alpha, p, w):
        return (1 - w) * mu

    def var(self, mu, alpha, p, w):
        dispersion_factor = 1 + alpha * mu ** (p - 1) + w * mu
        var = dispersion_factor * self.mean(mu, alpha, p, w)
        return var

    def _moment(self, n, mu, alpha, p, w):
        s, p = self.convert_params(mu, alpha, p)
        return (1 - w) * nbinom.moment(n, s, p)

    def convert_params(self, mu, alpha, p):
        size = 1.0 / alpha * mu ** (2 - p)
        prob = size / (size + mu)
        return (size, prob)