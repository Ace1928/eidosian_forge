import numpy as np
from numpy.linalg import lstsq
def _lowess_update_nn(x, cur_nn, i):
    """
    Update the endpoints of the nearest neighbors to
    the ith point.

    Parameters
    ----------
    x : iterable
        The sorted points of x-values
    cur_nn : list of length 2
        The two current indices between which are the
        k closest points to x[i]. (The actual value of
        k is irrelevant for the algorithm.
    i : int
        The index of the current value in x for which
        the k closest points are desired.

    Returns
    -------
    Nothing. It modifies cur_nn in place.
    """
    while True:
        if cur_nn[1] < x.size:
            left_dist = x[i] - x[cur_nn[0]]
            new_right_dist = x[cur_nn[1]] - x[i]
            if new_right_dist < left_dist:
                cur_nn[0] = cur_nn[0] + 1
                cur_nn[1] = cur_nn[1] + 1
            else:
                break
        else:
            break