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
def _check_func(checker, argname, thefunc, x0, args, numinputs, output_shape=None):
    res = atleast_1d(thefunc(*(x0[:numinputs],) + args))
    if output_shape is not None and shape(res) != output_shape:
        if output_shape[0] != 1:
            if len(output_shape) > 1:
                if output_shape[1] == 1:
                    return shape(res)
            msg = f"{checker}: there is a mismatch between the input and output shape of the '{argname}' argument"
            func_name = getattr(thefunc, '__name__', None)
            if func_name:
                msg += " '%s'." % func_name
            else:
                msg += '.'
            msg += f'Shape should be {output_shape} but it is {shape(res)}.'
            raise TypeError(msg)
    if issubdtype(res.dtype, inexact):
        dt = res.dtype
    else:
        dt = dtype(float)
    return (shape(res), dt)