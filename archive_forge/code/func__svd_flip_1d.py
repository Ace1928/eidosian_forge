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
def _svd_flip_1d(u, v):
    """Same as svd_flip but works on 1d arrays, and is inplace"""
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign