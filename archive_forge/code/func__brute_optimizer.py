from inspect import signature
import numpy as np
from scipy.optimize import brute, shgo
import pennylane as qml
def _brute_optimizer(fun, num_steps, bounds=None, **kwargs):
    """Brute force optimizer, wrapper of scipy.optimize.brute that repeats it
    ``num_steps`` times. Signature is as expected by ``RotosolveOptimizer._min_numeric``
    below, returning a scalar minimal position and the function value at that position."""
    Ns = kwargs.pop('Ns')
    width = bounds[0][1] - bounds[0][0]
    center = (bounds[0][1] + bounds[0][0]) / 2
    for _ in range(num_steps):
        range_ = (center - width / 2, center + width / 2)
        center, y_min, *_ = brute(fun, ranges=(range_,), full_output=True, Ns=Ns, **kwargs)
        center = center[0]
        width /= Ns
    return (center, y_min)