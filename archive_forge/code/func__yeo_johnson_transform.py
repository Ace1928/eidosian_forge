import warnings
from numbers import Integral, Real
import numpy as np
from scipy import optimize, sparse, stats
from scipy.special import boxcox
from ..base import (
from ..utils import _array_api, check_array
from ..utils._array_api import get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _incremental_mean_and_var, row_norms
from ..utils.sparsefuncs import (
from ..utils.sparsefuncs_fast import (
from ..utils.validation import (
from ._encoders import OneHotEncoder
def _yeo_johnson_transform(self, x, lmbda):
    """Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        """
    out = np.zeros_like(x)
    pos = x >= 0
    if abs(lmbda) < np.spacing(1.0):
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda
    if abs(lmbda - 2) > np.spacing(1.0):
        out[~pos] = -(np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
    else:
        out[~pos] = -np.log1p(-x[~pos])
    return out