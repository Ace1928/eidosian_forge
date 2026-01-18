import numpy as np
from . import _zeros_py as optzeros
from ._numdiff import approx_derivative
def _root_scalar_brenth_doc():
    """
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  `f(x, *args)` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass