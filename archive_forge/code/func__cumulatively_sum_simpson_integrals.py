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
def _cumulatively_sum_simpson_integrals(y: np.ndarray, dx: np.ndarray, integration_func: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """Calculate cumulative sum of Simpson integrals.
    Takes as input the integration function to be used. 
    The integration_func is assumed to return the cumulative sum using
    composite Simpson's rule. Assumes the axis of summation is -1.
    """
    sub_integrals_h1 = integration_func(y, dx)
    sub_integrals_h2 = integration_func(y[..., ::-1], dx[..., ::-1])[..., ::-1]
    shape = list(sub_integrals_h1.shape)
    shape[-1] += 1
    sub_integrals = np.empty(shape)
    sub_integrals[..., :-1:2] = sub_integrals_h1[..., ::2]
    sub_integrals[..., 1::2] = sub_integrals_h2[..., ::2]
    sub_integrals[..., -1] = sub_integrals_h2[..., -1]
    res = np.cumsum(sub_integrals, axis=-1)
    return res