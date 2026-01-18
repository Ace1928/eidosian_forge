import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.linalg import svd
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_consistent_length
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import svd_flip
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import FLOAT_DTYPES, check_is_fitted
def _get_first_singular_vectors_svd(X, Y):
    """Return the first left and right singular vectors of X'Y.

    Here the whole SVD is computed.
    """
    C = np.dot(X.T, Y)
    U, _, Vt = svd(C, full_matrices=False)
    return (U[:, 0], Vt[0, :])