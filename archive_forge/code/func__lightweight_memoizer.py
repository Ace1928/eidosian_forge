import warnings
from . import _minpack
import numpy as np
from numpy import (atleast_1d, triu, shape, transpose, zeros, prod, greater,
from scipy import linalg
from scipy.linalg import svd, cholesky, solve_triangular, LinAlgError
from scipy._lib._util import _asarray_validated, _lazywhere, _contains_nan
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._optimize import OptimizeResult, _check_unknown_options, OptimizeWarning
from ._lsq import least_squares
from ._lsq.least_squares import prepare_bounds
from scipy.optimize._minimize import Bounds
from numpy import dot, eye, take  # noqa: F401
from numpy.linalg import inv  # noqa: F401
def _lightweight_memoizer(f):

    def _memoized_func(params):
        if _memoized_func.skip_lookup:
            return f(params)
        if np.all(_memoized_func.last_params == params):
            return _memoized_func.last_val
        elif _memoized_func.last_params is not None:
            _memoized_func.skip_lookup = True
        val = f(params)
        if _memoized_func.last_params is None:
            _memoized_func.last_params = np.copy(params)
            _memoized_func.last_val = val
        return val
    _memoized_func.last_params = None
    _memoized_func.last_val = None
    _memoized_func.skip_lookup = False
    return _memoized_func