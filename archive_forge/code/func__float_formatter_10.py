import math
import warnings
import sys
import inspect
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
import numpy as np
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import MapWrapper, check_random_state
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
def _float_formatter_10(x):
    """
    Returns a string representation of a float with exactly ten characters
    """
    if np.isposinf(x):
        return '       inf'
    elif np.isneginf(x):
        return '      -inf'
    elif np.isnan(x):
        return '       nan'
    return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)