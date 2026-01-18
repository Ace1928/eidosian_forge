import warnings
import numpy
from numpy import (array, isfinite, inexact, nonzero, iscomplexobj,
from scipy._lib._util import _asarray_validated
from ._misc import LinAlgError, _datacopied, norm
from .lapack import get_lapack_funcs, _compute_lwork
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def _make_eigvals(alpha, beta, homogeneous_eigvals):
    if homogeneous_eigvals:
        if beta is None:
            return numpy.vstack((alpha, numpy.ones_like(alpha)))
        else:
            return numpy.vstack((alpha, beta))
    elif beta is None:
        return alpha
    else:
        w = numpy.empty_like(alpha)
        alpha_zero = alpha == 0
        beta_zero = beta == 0
        beta_nonzero = ~beta_zero
        w[beta_nonzero] = alpha[beta_nonzero] / beta[beta_nonzero]
        w[~alpha_zero & beta_zero] = numpy.inf
        if numpy.all(alpha.imag == 0):
            w[alpha_zero & beta_zero] = numpy.nan
        else:
            w[alpha_zero & beta_zero] = complex(numpy.nan, numpy.nan)
        return w