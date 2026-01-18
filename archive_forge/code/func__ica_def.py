import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import as_float_array, check_array, check_random_state
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.validation import check_is_fitted
def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """Deflationary FastICA using fun approx to neg-entropy function

    Used internally by FastICA.
    """
    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    n_iter = []
    for j in range(n_components):
        w = w_init[j, :].copy()
        w /= np.sqrt((w ** 2).sum())
        for i in range(max_iter):
            gwtx, g_wtx = g(np.dot(w.T, X), fun_args)
            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w
            _gs_decorrelation(w1, W, j)
            w1 /= np.sqrt((w1 ** 2).sum())
            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break
        n_iter.append(i + 1)
        W[j, :] = w
    return (W, max(n_iter))