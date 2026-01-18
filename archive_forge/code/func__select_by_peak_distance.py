import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _select_by_peak_distance(peaks, priority, distance):
    """
    Evaluate which peaks fulfill the distance condition.

    Parameters
    ----------
    peaks : ndarray
        Indices of peaks in `vector`.
    priority : ndarray
        An array matching `peaks` used to determine priority of each peak. A
        peak with a higher priority value is kept over one with a lower one.
    distance : np.float64
        Minimal distance that peaks must be spaced.

    Returns
    -------
    keep : ndarray[bool]
        A boolean mask evaluating to true where `peaks` fulfill the distance
        condition.

    Notes
    -----
    Declaring the input arrays as C-contiguous doesn't seem to have performance
    advantages.
    """
    peaks_size = peaks.shape[0]
    distance_ = cupy.ceil(distance)
    keep = cupy.ones(peaks_size, dtype=cupy.bool_)
    priority_to_position = cupy.argsort(priority)
    for i in range(peaks_size - 1, -1, -1):
        j = priority_to_position[i]
        if keep[j] == 0:
            continue
        k = j - 1
        while 0 <= k and peaks[j] - peaks[k] < distance_:
            keep[k] = 0
            k -= 1
        k = j + 1
        while k < peaks_size and peaks[k] - peaks[j] < distance_:
            keep[k] = 0
            k += 1
    return keep