import numpy as np
from numpy import poly1d, sqrt, exp
import scipy
from scipy import stats, special
from scipy.stats import distributions
from statsmodels.stats.moment_helpers import mvsk2mc, mc2mvsk
class SkewNorm2_gen(distributions.rv_continuous):
    """univariate Skew-Normal distribution of Azzalini

    class follows scipy.stats.distributions pattern

    """

    def _argcheck(self, alpha):
        return 1

    def _pdf(self, x, alpha):
        return 2.0 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2.0) * special.ndtr(alpha * x)