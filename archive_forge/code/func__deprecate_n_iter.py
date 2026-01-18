import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.linalg import pinvh
from ..base import RegressorMixin, _fit_context
from ..utils import _safe_indexing
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import fast_logdet
from ..utils.validation import _check_sample_weight
from ._base import LinearModel, _preprocess_data, _rescale_data
def _deprecate_n_iter(n_iter, max_iter):
    """Deprecates n_iter in favour of max_iter. Checks if the n_iter has been
    used instead of max_iter and generates a deprecation warning if True.

    Parameters
    ----------
    n_iter : int,
        Value of n_iter attribute passed by the estimator.

    max_iter : int, default=None
        Value of max_iter attribute passed by the estimator.
        If `None`, it corresponds to `max_iter=300`.

    Returns
    -------
    max_iter : int,
        Value of max_iter which shall further be used by the estimator.

    Notes
    -----
    This function should be completely removed in 1.5.
    """
    if n_iter != 'deprecated':
        if max_iter is not None:
            raise ValueError('Both `n_iter` and `max_iter` attributes were set. Attribute `n_iter` was deprecated in version 1.3 and will be removed in 1.5. To avoid this error, only set the `max_iter` attribute.')
        warnings.warn("'n_iter' was renamed to 'max_iter' in version 1.3 and will be removed in 1.5", FutureWarning)
        max_iter = n_iter
    elif max_iter is None:
        max_iter = 300
    return max_iter