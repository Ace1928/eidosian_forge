import numpy as np
from numpy import poly1d, sqrt, exp
import scipy
from scipy import stats, special
from scipy.stats import distributions
from statsmodels.stats.moment_helpers import mvsk2mc, mc2mvsk
class ACSkewT_gen(distributions.rv_continuous):
    """univariate Skew-T distribution of Azzalini

    class follows scipy.stats.distributions pattern
    but with __init__
    """

    def __init__(self):
        distributions.rv_continuous.__init__(self, name='Skew T distribution', shapes='df, alpha')

    def _argcheck(self, df, alpha):
        return (alpha == alpha) * (df > 0)

    def _rvs(self, df, alpha):
        V = stats.chi2.rvs(df, size=self._size)
        z = skewnorm.rvs(alpha, size=self._size)
        return z / np.sqrt(V / df)

    def _munp(self, n, df, alpha):
        return self._mom0_sc(n, df, alpha)

    def _pdf(self, x, df, alpha):
        return 2.0 * distributions.t._pdf(x, df) * special.stdtr(df + 1, alpha * x * np.sqrt((1 + df) / (x ** 2 + df)))