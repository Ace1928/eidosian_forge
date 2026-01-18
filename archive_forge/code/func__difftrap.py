from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import (_NoValue, _deprecate_positional_args,
def _difftrap(function, interval, numtraps):
    """
    Perform part of the trapezoidal rule to integrate a function.
    Assume that we had called difftrap with all lower powers-of-2
    starting with 1. Calling difftrap only returns the summation
    of the new ordinates. It does _not_ multiply by the width
    of the trapezoids. This must be performed by the caller.
        'function' is the function to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    if numtraps <= 0:
        raise ValueError('numtraps must be > 0 in difftrap().')
    elif numtraps == 1:
        return 0.5 * (function(interval[0]) + function(interval[1]))
    else:
        numtosum = numtraps / 2
        h = float(interval[1] - interval[0]) / numtosum
        lox = interval[0] + 0.5 * h
        points = lox + h * np.arange(numtosum)
        s = np.sum(function(points), axis=0)
        return s