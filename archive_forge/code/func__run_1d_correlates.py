import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def _run_1d_correlates(input, params, get_weights, output, mode, cval, origin=0):
    """
    Enhanced version of _run_1d_filters that uses correlate1d as the filter
    function. The params are a list of values to pass to the get_weights
    callable given. If duplicate param values are found, the weights are
    reused from the first invocation of get_weights. The get_weights callable
    must return a 1D array of weights to give to correlate1d.
    """
    wghts = {}
    for param in params:
        if param not in wghts:
            wghts[param] = get_weights(param)
    wghts = [wghts[param] for param in params]
    return _filters_core._run_1d_filters([None if w is None else correlate1d for w in wghts], input, wghts, output, mode, cval, origin)