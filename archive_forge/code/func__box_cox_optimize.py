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
def _box_cox_optimize(self, x):
    """Find and return optimal lambda parameter of the Box-Cox transform by
        MLE, for observed data x.

        We here use scipy builtins which uses the brent optimizer.
        """
    mask = np.isnan(x)
    if np.all(mask):
        raise ValueError('Column must not be all nan.')
    _, lmbda = stats.boxcox(x[~mask], lmbda=None)
    return lmbda