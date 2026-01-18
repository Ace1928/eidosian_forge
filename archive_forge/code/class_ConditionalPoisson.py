import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
import collections
import warnings
import itertools
class ConditionalPoisson(_ConditionalModel):
    """
    Fit a conditional Poisson regression model to grouped data.

    Every group is implicitly given an intercept, but the model is fit using
    a conditional likelihood in which the intercepts are not present.  Thus,
    intercept estimates are not given, but the other parameter estimates can
    be interpreted as being adjusted for any group-level confounders.

    Parameters
    ----------
    endog : array_like
        The response variable
    exog : array_like
        The covariates
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.
    """

    def loglike(self, params):
        ofs = None
        if hasattr(self, 'offset'):
            ofs = self._offset_grp
        ll = 0.0
        for i in range(len(self._endog_grp)):
            xb = np.dot(self._exog_grp[i], params)
            if ofs is not None:
                xb += ofs[i]
            exb = np.exp(xb)
            y = self._endog_grp[i]
            ll += np.dot(y, xb)
            s = exb.sum()
            ll -= self._sumy[i] * np.log(s)
        return ll

    def score(self, params):
        ofs = None
        if hasattr(self, 'offset'):
            ofs = self._offset_grp
        score = 0.0
        for i in range(len(self._endog_grp)):
            x = self._exog_grp[i]
            xb = np.dot(x, params)
            if ofs is not None:
                xb += ofs[i]
            exb = np.exp(xb)
            s = exb.sum()
            y = self._endog_grp[i]
            score += np.dot(y, x)
            score -= self._sumy[i] * np.dot(exb, x) / s
        return score