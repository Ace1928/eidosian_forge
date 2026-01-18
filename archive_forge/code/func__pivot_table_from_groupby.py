import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
@staticmethod
def _pivot_table_from_groupby(df, dropna, drop_column_level, to_unstack, fill_value, sort=False):
    """
        Convert group by aggregation result to a pivot table.

        Parameters
        ----------
        df : pandas.DataFrame
            Group by aggregation result.
        dropna : bool
            Whether to drop NaN columns.
        drop_column_level : int or None
            An extra columns level to drop.
        to_unstack : list of labels or None
            Group by keys to pass to ``.unstack()``. Reperent `columns` parameter
            for ``.pivot_table()``.
        fill_value : bool
            Fill value for NaN values.
        sort : bool, default: False
            Whether to sort the result along index.

        Returns
        -------
        pandas.DataFrame
        """
    if df.index.nlevels > 1 and to_unstack is not None:
        df = df.unstack(level=to_unstack)
    if drop_column_level is not None:
        df = df.droplevel(drop_column_level, axis=1)
    if dropna:
        df = df.dropna(axis=1, how='all')
    if fill_value is not None:
        df = df.fillna(fill_value, downcast='infer')
    if sort:
        df = df.sort_index(axis=0)
    return df