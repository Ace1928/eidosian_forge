import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats.distributions import rv_continuous
import matplotlib.pyplot as plt
from numpy import where, inf
from numpy import abs as np_abs
class genpareto2_gen(rv_continuous):

    def _argcheck(self, c):
        c = np.asarray(c)
        self.b = where(c > 0, 1.0 / np_abs(c), inf)
        return where(c == 0, 0, 1)

    def _pdf(self, x, c):
        Px = np.power(1 - c * x, -1.0 + 1.0 / c)
        return Px

    def _logpdf(self, x, c):
        return (-1.0 + 1.0 / c) * np.log1p(-c * x)

    def _cdf(self, x, c):
        return 1.0 - np.power(1 - c * x, 1.0 / c)

    def _ppf(self, q, c):
        vals = -1.0 / c * (np.power(1 - q, c) - 1)
        return vals

    def _munp(self, n, c):
        k = np.arange(0, n + 1)
        val = (1.0 / c) ** n * np.sum(comb(n, k) * (-1) ** k / (1.0 + c * k), axis=0)
        return where(c * n > -1, val, inf)

    def _entropy(self, c):
        if c < 0:
            return 1 - c
        else:
            self.b = 1.0 / c
            return rv_continuous._entropy(self, c)