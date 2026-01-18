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
def _recover_from_bracket_error(solver, fun, bracket, args, **options):
    try:
        res = solver(fun, bracket, args, **options)
    except BracketError as e:
        msg = str(e)
        xa, xb, xc, fa, fb, fc, funcalls = e.data
        xs, fs = ([xa, xb, xc], [fa, fb, fc])
        if np.any(np.isnan([xs, fs])):
            x, fun = (np.nan, np.nan)
        else:
            imin = np.argmin(fs)
            x, fun = (xs[imin], fs[imin])
        return OptimizeResult(fun=fun, nfev=funcalls, x=x, nit=0, success=False, message=msg)
    return res