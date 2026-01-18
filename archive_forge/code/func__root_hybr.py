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
def _root_hybr(func, x0, args=(), jac=None, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, eps=None, factor=100, diag=None, **unknown_options):
    """
    Find the roots of a multivariate function using MINPACK's hybrd and
    hybrj routines (modified Powell method).

    Options
    -------
    col_deriv : bool
        Specify whether the Jacobian function computes derivatives down
        the columns (faster, because there is no transpose operation).
    xtol : float
        The calculation will terminate if the relative error between two
        consecutive iterates is at most `xtol`.
    maxfev : int
        The maximum number of calls to the function. If zero, then
        ``100*(N+1)`` is the maximum where N is the number of elements
        in `x0`.
    band : tuple
        If set to a two-sequence containing the number of sub- and
        super-diagonals within the band of the Jacobi matrix, the
        Jacobi matrix is considered banded (only for ``fprime=None``).
    eps : float
        A suitable step length for the forward-difference
        approximation of the Jacobian (for ``fprime=None``). If
        `eps` is less than the machine precision, it is assumed
        that the relative errors in the functions are of the order of
        the machine precision.
    factor : float
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in the interval
        ``(0.1, 100)``.
    diag : sequence
        N positive entries that serve as a scale factors for the
        variables.

    """
    _check_unknown_options(unknown_options)
    epsfcn = eps
    x0 = asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple):
        args = (args,)
    shape, dtype = _check_func('fsolve', 'func', func, x0, args, n, (n,))
    if epsfcn is None:
        epsfcn = finfo(dtype).eps
    Dfun = jac
    if Dfun is None:
        if band is None:
            ml, mu = (-10, -10)
        else:
            ml, mu = band[:2]
        if maxfev == 0:
            maxfev = 200 * (n + 1)
        retval = _minpack._hybrd(func, x0, args, 1, xtol, maxfev, ml, mu, epsfcn, factor, diag)
    else:
        _check_func('fsolve', 'fprime', Dfun, x0, args, n, (n, n))
        if maxfev == 0:
            maxfev = 100 * (n + 1)
        retval = _minpack._hybrj(func, Dfun, x0, args, 1, col_deriv, xtol, maxfev, factor, diag)
    x, status = (retval[0], retval[-1])
    errors = {0: 'Improper input parameters were entered.', 1: 'The solution converged.', 2: 'The number of calls to function has reached maxfev = %d.' % maxfev, 3: 'xtol=%f is too small, no further improvement in the approximate\n  solution is possible.' % xtol, 4: 'The iteration is not making good progress, as measured by the \n  improvement from the last five Jacobian evaluations.', 5: 'The iteration is not making good progress, as measured by the \n  improvement from the last ten iterations.', 'unknown': 'An error occurred.'}
    info = retval[1]
    info['fun'] = info.pop('fvec')
    sol = OptimizeResult(x=x, success=status == 1, status=status, method='hybr')
    sol.update(info)
    try:
        sol['message'] = errors[status]
    except KeyError:
        sol['message'] = errors['unknown']
    return sol