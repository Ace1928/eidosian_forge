import numpy as np
from scipy.optimize._slsqp import slsqp
from numpy import (zeros, array, linalg, append, concatenate, finfo,
from ._optimize import (OptimizeResult, _check_unknown_options,
from ._numdiff import approx_derivative
from ._constraints import old_bound_to_new, _arr_to_scalar
from scipy._lib._array_api import atleast_nd, array_namespace
from numpy import exp, inf  # noqa: F401

    Minimize a scalar function of one or more variables using Sequential
    Least Squares Programming (SLSQP).

    Options
    -------
    ftol : float
        Precision goal for the value of f in the stopping criterion.
    eps : float
        Step size used for numerical approximation of the Jacobian.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored and set to 0.
    maxiter : int
        Maximum number of iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of `jac`. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    