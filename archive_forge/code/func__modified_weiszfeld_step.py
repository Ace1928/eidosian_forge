import warnings
from itertools import combinations
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import binom
from ..base import RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..utils import check_random_state
from ..utils._param_validation import Interval
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel
def _modified_weiszfeld_step(X, x_old):
    """Modified Weiszfeld step.

    This function defines one iteration step in order to approximate the
    spatial median (L1 median). It is a form of an iteratively re-weighted
    least squares method.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    x_old : ndarray of shape = (n_features,)
        Current start vector.

    Returns
    -------
    x_new : ndarray of shape (n_features,)
        New iteration step.

    References
    ----------
    - On Computation of Spatial Median for Robust Data Mining, 2005
      T. Kärkkäinen and S. Äyrämö
      http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf
    """
    diff = X - x_old
    diff_norm = np.sqrt(np.sum(diff ** 2, axis=1))
    mask = diff_norm >= _EPSILON
    is_x_old_in_X = int(mask.sum() < X.shape[0])
    diff = diff[mask]
    diff_norm = diff_norm[mask][:, np.newaxis]
    quotient_norm = linalg.norm(np.sum(diff / diff_norm, axis=0))
    if quotient_norm > _EPSILON:
        new_direction = np.sum(X[mask, :] / diff_norm, axis=0) / np.sum(1 / diff_norm, axis=0)
    else:
        new_direction = 1.0
        quotient_norm = 1.0
    return max(0.0, 1.0 - is_x_old_in_X / quotient_norm) * new_direction + min(1.0, is_x_old_in_X / quotient_norm) * x_old