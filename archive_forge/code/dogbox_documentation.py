import numpy as np
from numpy.linalg import lstsq, norm
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr
from scipy.optimize import OptimizeResult
from .common import (
Find dogleg step in a rectangular region.

    Returns
    -------
    step : ndarray, shape (n,)
        Computed dogleg step.
    bound_hits : ndarray of int, shape (n,)
        Each component shows whether a corresponding variable hits the
        initial bound after the step is taken:
            *  0 - a variable doesn't hit the bound.
            * -1 - lower bound is hit.
            *  1 - upper bound is hit.
    tr_hit : bool
        Whether the step hit the boundary of the trust-region.
    