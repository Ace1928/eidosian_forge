import numpy as np
from numpy import poly1d, sqrt, exp
import scipy
from scipy import stats, special
from scipy.stats import distributions
from statsmodels.stats.moment_helpers import mvsk2mc, mc2mvsk
class SkewNorm_gen(distributions.rv_continuous):
    """univariate Skew-Normal distribution of Azzalini

    class follows scipy.stats.distributions pattern
    but with __init__


    """

    def __init__(self):
        distributions.rv_continuous.__init__(self, name='Skew Normal distribution', shapes='alpha')

    def _argcheck(self, alpha):
        return 1

    def _rvs(self, alpha):
        delta = alpha / np.sqrt(1 + alpha ** 2)
        u0 = stats.norm.rvs(size=self._size)
        u1 = delta * u0 + np.sqrt(1 - delta ** 2) * stats.norm.rvs(size=self._size)
        return np.where(u0 > 0, u1, -u1)

    def _munp(self, n, alpha):
        return self._mom0_sc(n, alpha)

    def _pdf(self, x, alpha):
        return 2.0 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2.0) * special.ndtr(alpha * x)

    def _stats_skip(self, x, alpha, moments='mvsk'):
        pass