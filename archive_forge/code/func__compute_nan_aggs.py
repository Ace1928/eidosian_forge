from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@staticmethod
def _compute_nan_aggs(raw_df: np.ndarray, cols: pandas.Index, nan_mask: np.ndarray) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
        Compute sums, sums of square and the number of observations for a partition assuming there are NaN values in it.

        Parameters
        ----------
        raw_df : np.ndarray
            Raw values of the partition to compute the aggregations for.
        cols : pandas.Index
            Columns of the partition.
        nan_mask : np.ndarray[bool]
            Boolean mask showing positions of NaN values in the `raw_df`.

        Returns
        -------
        Tuple[sums: pandas.DataFrame, sums_of_squares: pandas.DataFrame, count: pandas.DataFrame]
            A tuple storing DataFrames where each of them holds the result for
            one of the described aggregations.
        """
    sums = {}
    sums_of_squares = {}
    count = {}
    for i, col in enumerate(cols):
        col_vals = np.resize(raw_df[i], raw_df.shape)
        np.putmask(col_vals, nan_mask, values=0)
        sums[col] = pandas.Series(np.sum(col_vals, axis=1), index=cols, copy=False)
        sums_of_squares[col] = pandas.Series(np.sum(col_vals ** 2, axis=1), index=cols, copy=False)
        count[col] = pandas.Series(nan_mask.shape[1] - np.count_nonzero(nan_mask | nan_mask[i], axis=1), index=cols, copy=False)
    sums = pandas.concat(sums, axis=1, copy=False)
    sums_of_squares = pandas.concat(sums_of_squares, axis=1, copy=False)
    count = pandas.concat(count, axis=1, copy=False)
    return (sums, sums_of_squares, count)