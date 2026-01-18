import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import as_float_array, check_array, check_random_state
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.validation import check_is_fitted
def _cube(x, fun_args):
    return (x ** 3, (3 * x ** 2).mean(axis=-1))