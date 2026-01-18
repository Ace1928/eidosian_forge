import numpy as np
import numba
from .exceptions import ParameterError
from .utils import valid_intervals
from .._typing import _SequenceLike
@numba.jit(nopython=True, cache=True)
def __jaccard(int_a: np.ndarray, int_b: np.ndarray):
    """Jaccard similarity between two intervals

    Parameters
    ----------
    int_a, int_b : np.ndarrays, shape=(2,)

    Returns
    -------
    Jaccard similarity between intervals
    """
    ends = [int_a[1], int_b[1]]
    if ends[1] < ends[0]:
        ends.reverse()
    starts = [int_a[0], int_b[0]]
    if starts[1] < starts[0]:
        starts.reverse()
    intersection = ends[0] - starts[1]
    if intersection < 0:
        intersection = 0.0
    union = ends[1] - starts[0]
    if union > 0:
        return intersection / union
    return 0.0