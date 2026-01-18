import numpy as np
from . import _zeros_py as optzeros
from ._numdiff import approx_derivative
def _root_scalar_secant_doc():
    """
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    x1 : float, required
        A second guess.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass