import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def _gen_ridge_line(start_locs, max_locs, length, distances, gaps):
    """
    Generate coordinates for a ridge line.

    Will be a series of coordinates, starting a start_loc (length 2).
    The maximum distance between any adjacent columns will be
    `max_distance`, the max distance between adjacent rows
    will be `map_gap'.

    `max_locs` should be the size of the intended matrix. The
    ending coordinates are guaranteed to be less than `max_locs`,
    although they may not approach `max_locs` at all.
    """

    def keep_bounds(num, max_val):
        out = max(num, 0)
        out = min(out, max_val)
        return out
    gaps = copy.deepcopy(gaps)
    distances = copy.deepcopy(distances)
    locs = np.zeros([length, 2], dtype=int)
    locs[0, :] = start_locs
    total_length = max_locs[0] - start_locs[0] - sum(gaps)
    if total_length < length:
        raise ValueError('Cannot generate ridge line according to constraints')
    dist_int = length / len(distances) - 1
    gap_int = length / len(gaps) - 1
    for ind in range(1, length):
        nextcol = locs[ind - 1, 1]
        nextrow = locs[ind - 1, 0] + 1
        if ind % dist_int == 0 and len(distances) > 0:
            nextcol += (-1) ** ind * distances.pop()
        if ind % gap_int == 0 and len(gaps) > 0:
            nextrow += gaps.pop()
        nextrow = keep_bounds(nextrow, max_locs[0])
        nextcol = keep_bounds(nextcol, max_locs[1])
        locs[ind, :] = [nextrow, nextcol]
    return [locs[:, 0], locs[:, 1]]