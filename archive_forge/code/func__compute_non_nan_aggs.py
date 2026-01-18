from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@staticmethod
def _compute_non_nan_aggs(df: pandas.DataFrame) -> Tuple[pandas.Series, pandas.Series, pandas.Series]:
    """
        Compute sums, sums of square and the number of observations for a partition assuming there are no NaN values in it.

        Parameters
        ----------
        df : pandas.DataFrame
            Partition to compute the aggregations for.

        Returns
        -------
        Tuple[sums: pandas.Series, sums_of_squares: pandas.Series, count: pandas.Series]
            A tuple storing Series where each of them holds the result for
            one of the described aggregations.
        """
    sums = df.sum().rename(MODIN_UNNAMED_SERIES_LABEL)
    sums_of_squares = (df ** 2).sum().rename(MODIN_UNNAMED_SERIES_LABEL)
    count = pandas.Series(np.repeat(len(df), len(df.columns)), index=df.columns, copy=False).rename(MODIN_UNNAMED_SERIES_LABEL)
    return (sums, sums_of_squares, count)