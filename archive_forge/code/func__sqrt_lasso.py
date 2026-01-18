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
def _sqrt_lasso(self, alpha, refit, zero_tol):
    try:
        import cvxopt
    except ImportError:
        msg = 'sqrt_lasso fitting requires the cvxopt module'
        raise ValueError(msg)
    n = len(self.endog)
    p = self.exog.shape[1]
    h0 = cvxopt.matrix(0.0, (2 * p + 1, 1))
    h1 = cvxopt.matrix(0.0, (n + 1, 1))
    h1[1:, 0] = cvxopt.matrix(self.endog, (n, 1))
    G0 = cvxopt.spmatrix([], [], [], (2 * p + 1, 2 * p + 1))
    for i in range(1, 2 * p + 1):
        G0[i, i] = -1
    G1 = cvxopt.matrix(0.0, (n + 1, 2 * p + 1))
    G1[0, 0] = -1
    G1[1:, 1:p + 1] = self.exog
    G1[1:, p + 1:] = -self.exog
    c = cvxopt.matrix(alpha / n, (2 * p + 1, 1))
    c[0] = 1 / np.sqrt(n)
    from cvxopt import solvers
    solvers.options['show_progress'] = False
    rslt = solvers.socp(c, Gl=G0, hl=h0, Gq=[G1], hq=[h1])
    x = np.asarray(rslt['x']).flat
    bp = x[1:p + 1]
    bn = x[p + 1:]
    params = bp - bn
    if not refit:
        return params
    ii = np.flatnonzero(np.abs(params) > zero_tol)
    rfr = OLS(self.endog, self.exog[:, ii]).fit()
    params *= 0
    params[ii] = rfr.params
    return params