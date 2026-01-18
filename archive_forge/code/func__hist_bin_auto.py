import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def _hist_bin_auto(x, range):
    """
    Histogram bin estimator that uses the minimum width of the
    Freedman-Diaconis and Sturges estimators if the FD bin width is non-zero.
    If the bin width from the FD estimator is 0, the Sturges estimator is used.

    The FD estimator is usually the most robust method, but its width
    estimate tends to be too large for small `x` and bad for data with limited
    variance. The Sturges estimator is quite good for small (<1000) datasets
    and is the default in the R language. This method gives good off-the-shelf
    behaviour.

    .. versionchanged:: 1.15.0
    If there is limited variance the IQR can be 0, which results in the
    FD bin width being 0 too. This is not a valid bin width, so
    ``np.histogram_bin_edges`` chooses 1 bin instead, which may not be optimal.
    If the IQR is 0, it's unlikely any variance-based estimators will be of
    use, so we revert to the Sturges estimator, which only uses the size of the
    dataset in its calculation.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.

    See Also
    --------
    _hist_bin_fd, _hist_bin_sturges
    """
    fd_bw = _hist_bin_fd(x, range)
    sturges_bw = _hist_bin_sturges(x, range)
    del range
    if fd_bw:
        return min(fd_bw, sturges_bw)
    else:
        return sturges_bw