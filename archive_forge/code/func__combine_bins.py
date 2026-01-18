import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
def _combine_bins(edge_index, x):
    """group columns into bins using sum

    This is mainly a helper function for combining probabilities into cells.
    It similar to `np.add.reduceat(x, edge_index, axis=-1)` except for the
    treatment of the last index and last cell.

    Parameters
    ----------
    edge_index : array_like
         This defines the (zero-based) indices for the columns that are be
         combined. Each index in `edge_index` except the last is the starting
         index for a bin. The largest index in a bin is the next edge_index-1.
    x : 1d or 2d array
        array for which columns are combined. If x is 1-dimensional that it
        will be treated as a 2-d row vector.

    Returns
    -------
    x_new : ndarray
    k_li : ndarray
        Count of columns combined in bin.


    Examples
    --------
    >>> dia.combine_bins([0,1,5], np.arange(4))
    (array([0, 6]), array([1, 4]))

    this aggregates to two bins with the sum of 1 and 4 elements
    >>> np.arange(4)[0].sum()
    0
    >>> np.arange(4)[1:5].sum()
    6

    If the rightmost index is smaller than len(x)+1, then the remaining
    columns will not be included.

    >>> dia.combine_bins([0,1,3], np.arange(4))
    (array([0, 3]), array([1, 2]))
    """
    x = np.asarray(x)
    if x.ndim == 1:
        is_1d = True
        x = x[None, :]
    else:
        is_1d = False
    xli = []
    kli = []
    for bin_idx in range(len(edge_index) - 1):
        i, j = edge_index[bin_idx:bin_idx + 2]
        xli.append(x[:, i:j].sum(1))
        kli.append(j - i)
    x_new = np.column_stack(xli)
    if is_1d:
        x_new = x_new.squeeze()
    return (x_new, np.asarray(kli))