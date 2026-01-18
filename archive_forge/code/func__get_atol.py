import warnings
from textwrap import dedent
import numpy as np
from . import _iterative
from scipy.sparse.linalg._interface import LinearOperator
from .utils import make_system
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import non_reentrant
def _get_atol(tol, atol, bnrm2, get_residual, routine_name):
    """
    Parse arguments for absolute tolerance in termination condition.

    Parameters
    ----------
    tol, atol : object
        The arguments passed into the solver routine by user.
    bnrm2 : float
        2-norm of the rhs vector.
    get_residual : callable
        Callable ``get_residual()`` that returns the initial value of
        the residual.
    routine_name : str
        Name of the routine.
    """
    if atol is None:
        warnings.warn("scipy.sparse.linalg.{name} called without specifying `atol`. The default value will be changed in a future release. For compatibility, specify a value for `atol` explicitly, e.g., ``{name}(..., atol=0)``, or to retain the old behavior ``{name}(..., atol='legacy')``".format(name=routine_name), category=DeprecationWarning, stacklevel=4)
        atol = 'legacy'
    tol = float(tol)
    if atol == 'legacy':
        resid = get_residual()
        if resid <= tol:
            return 'exit'
        if bnrm2 == 0:
            return tol
        else:
            return tol * float(bnrm2)
    else:
        return max(float(atol), tol * float(bnrm2))