import warnings
import cupy
from cupy_backends.cuda.api import runtime
import cupyx.scipy.signal._signaltools as filtering
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal.windows._windows import get_window
def _median_bias(n):
    """
    Returns the bias of the median of a set of periodograms relative to
    the mean.

    See arXiv:gr-qc/0509116 Appendix B for details.

    Parameters
    ----------
    n : int
        Numbers of periodograms being averaged.

    Returns
    -------
    bias : float
        Calculated bias.
    """
    ii_2 = 2 * cupy.arange(1.0, (n - 1) // 2 + 1)
    return 1 + cupy.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)