from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
def _fit_ridge(self, alpha):
    """
        Fit a linear model using ridge regression.

        Parameters
        ----------
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.

        Notes
        -----
        Equivalent to fit_regularized with L1_wt = 0 (but implemented
        more efficiently).
        """
    u, s, vt = np.linalg.svd(self.exog, 0)
    v = vt.T
    q = np.dot(u.T, self.endog) * s
    s2 = s * s
    if np.isscalar(alpha):
        sd = s2 + alpha * self.nobs
        params = q / sd
        params = np.dot(v, params)
    else:
        alpha = np.asarray(alpha)
        vtav = self.nobs * np.dot(vt, alpha[:, None] * v)
        d = np.diag(vtav) + s2
        np.fill_diagonal(vtav, d)
        r = np.linalg.solve(vtav, q)
        params = np.dot(v, r)
    from statsmodels.base.elastic_net import RegularizedResults
    return RegularizedResults(self, params)