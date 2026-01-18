import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import as_float_array, check_array, check_random_state
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.validation import check_is_fitted
def _sym_decorrelation(W):
    """Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = linalg.eigh(np.dot(W, W.T))
    s = np.clip(s, a_min=np.finfo(W.dtype).tiny, a_max=None)
    return np.linalg.multi_dot([u * (1.0 / np.sqrt(s)), u.T, W])