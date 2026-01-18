import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
import collections
import warnings
import itertools
class ConditionalLogit(_ConditionalModel):
    """
    Fit a conditional logistic regression model to grouped data.

    Every group is implicitly given an intercept, but the model is fit using
    a conditional likelihood in which the intercepts are not present.  Thus,
    intercept estimates are not given, but the other parameter estimates can
    be interpreted as being adjusted for any group-level confounders.

    Parameters
    ----------
    endog : array_like
        The response variable, must contain only 0 and 1.
    exog : array_like
        The array of covariates.  Do not include an intercept
        in this array.
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.
    """

    def __init__(self, endog, exog, missing='none', **kwargs):
        super().__init__(endog, exog, missing=missing, **kwargs)
        if np.any(np.unique(self.endog) != np.r_[0, 1]):
            msg = 'endog must be coded as 0, 1'
            raise ValueError(msg)
        self.K = self.exog.shape[1]

    def loglike(self, params):
        ll = 0
        for g in range(len(self._endog_grp)):
            ll += self.loglike_grp(g, params)
        return ll

    def score(self, params):
        score = 0
        for g in range(self._n_groups):
            score += self.score_grp(g, params)
        return score

    def _denom(self, grp, params, ofs=None):
        if ofs is None:
            ofs = 0
        exb = np.exp(np.dot(self._exog_grp[grp], params) + ofs)
        memo = {}

        def f(t, k):
            if t < k:
                return 0
            if k == 0:
                return 1
            try:
                return memo[t, k]
            except KeyError:
                pass
            v = f(t - 1, k) + f(t - 1, k - 1) * exb[t - 1]
            memo[t, k] = v
            return v
        return f(self._groupsize[grp], self._n1[grp])

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

    def loglike_grp(self, grp, params):
        ofs = None
        if hasattr(self, 'offset'):
            ofs = self._offset_grp[grp]
        llg = np.dot(self._xy[grp], params)
        if ofs is not None:
            llg += self._endofs[grp]
        llg -= np.log(self._denom(grp, params, ofs))
        return llg

    def score_grp(self, grp, params):
        ofs = 0
        if hasattr(self, 'offset'):
            ofs = self._offset_grp[grp]
        d, h = self._denom_grad(grp, params, ofs)
        return self._xy[grp] - h / d