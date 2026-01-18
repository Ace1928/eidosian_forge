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
def _cumulative_simpson_equal_intervals(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Calculate the Simpson integrals for all h1 intervals assuming equal interval
    widths. The function can also be used to calculate the integral for all
    h2 intervals by reversing the inputs, `y` and `dx`.
    """
    d = dx[..., :-1]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]
    return d / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4)