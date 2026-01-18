import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
import collections
import warnings
import itertools
def _denom_grad(self, grp, params, ofs=None):
    if ofs is None:
        ofs = 0
    ex = self._exog_grp[grp]
    exb = np.exp(np.dot(ex, params) + ofs)
    memo = {}

    def s(t, k):
        if t < k:
            return (0, np.zeros(self.k_params))
        if k == 0:
            return (1, 0)
        try:
            return memo[t, k]
        except KeyError:
            pass
        h = exb[t - 1]
        a, b = s(t - 1, k)
        c, e = s(t - 1, k - 1)
        d = c * h * ex[t - 1, :]
        u, v = (a + c * h, b + d + e * h)
        memo[t, k] = (u, v)
        return (u, v)
    return s(self._groupsize[grp], self._n1[grp])