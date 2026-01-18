import functools
from threading import RLock
import numpy as np
from scipy.optimize import _cobyla as cobyla
from ._optimize import (OptimizeResult, _check_unknown_options,

    Minimize a scalar function of one or more variables using the
    Constrained Optimization BY Linear Approximation (COBYLA) algorithm.

    Options
    -------
    rhobeg : float
        Reasonable initial changes to the variables.
    tol : float
        Final accuracy in the optimization (not precisely guaranteed).
        This is a lower bound on the size of the trust region.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored as set to 0.
    maxiter : int
        Maximum number of function evaluations.
    catol : float
        Tolerance (absolute) for constraint violations

    