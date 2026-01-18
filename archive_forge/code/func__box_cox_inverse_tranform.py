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
def _box_cox_inverse_tranform(self, x, lmbda):
    """Return inverse-transformed input x following Box-Cox inverse
        transform with parameter lambda.
        """
    if lmbda == 0:
        x_inv = np.exp(x)
    else:
        x_inv = (x * lmbda + 1) ** (1 / lmbda)
    return x_inv