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
def _wrap_scalar_function(function, args):
    ncalls = [0]
    if function is None:
        return (ncalls, None)

    def function_wrapper(x, *wrapper_args):
        ncalls[0] += 1
        fx = function(np.copy(x), *wrapper_args + args)
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError('The user-provided objective function must return a scalar value.') from e
        return fx
    return (ncalls, function_wrapper)