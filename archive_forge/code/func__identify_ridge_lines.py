import math
import numpy as np
from scipy.signal._wavelets import _cwt, _ricker
from scipy.stats import scoreatpercentile
from ._peak_finding_utils import (
def _identify_ridge_lines(matr, max_distances, gap_thresh):
    """
    Identify ridges in the 2-D matrix.

    Expect that the width of the wavelet feature increases with increasing row
    number.

    Parameters
    ----------
    matr : 2-D ndarray
        Matrix in which to identify ridge lines.
    max_distances : 1-D sequence
        At each row, a ridge line is only connected
        if the relative max at row[n] is within
        `max_distances`[n] from the relative max at row[n+1].
    gap_thresh : int
        If a relative maximum is not found within `max_distances`,
        there will be a gap. A ridge line is discontinued if
        there are more than `gap_thresh` points without connecting
        a new relative maximum.

    Returns
    -------
    ridge_lines : tuple
        Tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the
        ii-th ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none
        found.  Each ridge-line will be sorted by row (increasing), but the
        order of the ridge lines is not specified.

    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
       :doi:`10.1093/bioinformatics/btl355`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal._peak_finding import _identify_ridge_lines
    >>> rng = np.random.default_rng()
    >>> data = rng.random((5,5))
    >>> max_dist = 3
    >>> max_distances = np.full(20, max_dist)
    >>> ridge_lines = _identify_ridge_lines(data, max_distances, 1)

    Notes
    -----
    This function is intended to be used in conjunction with `cwt`
    as part of `find_peaks_cwt`.

    """
    if len(max_distances) < matr.shape[0]:
        raise ValueError('Max_distances must have at least as many rows as matr')
    all_max_cols = _boolrelextrema(matr, np.greater, axis=1, order=1)
    has_relmax = np.nonzero(all_max_cols.any(axis=1))[0]
    if len(has_relmax) == 0:
        return []
    start_row = has_relmax[-1]
    ridge_lines = [[[start_row], [col], 0] for col in np.nonzero(all_max_cols[start_row])[0]]
    final_lines = []
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    for row in rows:
        this_max_cols = cols[all_max_cols[row]]
        for line in ridge_lines:
            line[2] += 1
        prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
        for ind, col in enumerate(this_max_cols):
            line = None
            if len(prev_ridge_cols) > 0:
                diffs = np.abs(col - prev_ridge_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = ridge_lines[closest]
            if line is not None:
                line[1].append(col)
                line[0].append(row)
                line[2] = 0
            else:
                new_line = [[row], [col], 0]
                ridge_lines.append(new_line)
        for ind in range(len(ridge_lines) - 1, -1, -1):
            line = ridge_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del ridge_lines[ind]
    out_lines = []
    for line in final_lines + ridge_lines:
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = (np.zeros_like(sortargs), np.zeros_like(sortargs))
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])
    return out_lines