import abc
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._libs.tslibs import to_offset
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype
from pandas.core.resample import _get_timestamp_range_edges
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
@_inherit_docstrings(ShuffleFunctions)
class ShuffleSortFunctions(ShuffleFunctions):
    """
    Perform the sampling, quantiles picking, and the splitting stages for the range-partitioning building.

    Parameters
    ----------
    modin_frame : PandasDataframe
        The frame to build the range-partitioning for.
    columns : str, list of strings or None
        The column/columns to use as a key. Can't be specified along with `level`.
    ascending : bool
        Whether the ranges should be in ascending or descending order.
    ideal_num_new_partitions : int
        The ideal number of new partitions.
    level : list of strings or ints, or None
        Index level(s) to use as a key. Can't be specified along with `columns`.
    closed_on_right : bool, default: False
        Whether to include the right limit in range-partitioning.
            True:  bins[i - 1] < x <= bins[i]
            False: bins[i - 1] <= x < bins[i]
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, modin_frame: 'PandasDataframe', columns: Optional[Union[str, list]], ascending: Union[list, bool], ideal_num_new_partitions: int, level: Optional[list[Union[str, int]]]=None, closed_on_right: bool=False, **kwargs: dict):
        self.frame_len = len(modin_frame)
        self.ideal_num_new_partitions = ideal_num_new_partitions
        self.columns = columns if is_list_like(columns) else [columns]
        self.ascending = ascending
        self.kwargs = kwargs.copy()
        self.level = level
        self.columns_info = None
        self.closed_on_right = closed_on_right

    def sample_fn(self, partition: pandas.DataFrame) -> pandas.DataFrame:
        if self.level is not None:
            partition = self._index_to_df_zero_copy(partition, self.level)
        else:
            partition = partition[self.columns]
        return self.pick_samples_for_quantiles(partition, self.ideal_num_new_partitions, self.frame_len)

    def pivot_fn(self, samples: 'list[pandas.DataFrame]') -> int:
        key = self.kwargs.get('key', None)
        samples = pandas.concat(samples, axis=0, copy=False)
        columns_info: 'list[ColumnInfo]' = []
        number_of_groups = 1
        cols = []
        for i, col in enumerate(samples.columns):
            num_pivots = int(self.ideal_num_new_partitions / number_of_groups)
            if num_pivots < 2 and len(columns_info):
                break
            column_val = samples[col]
            cols.append(col)
            is_numeric = is_numeric_dtype(column_val.dtype)
            method = 'linear' if is_numeric else 'inverted_cdf'
            pivots = self.pick_pivots_from_samples_for_sort(column_val, num_pivots, method, key)
            columns_info.append(ColumnInfo(self.level[i] if self.level is not None else col, pivots, is_numeric))
            number_of_groups *= len(pivots) + 1
        self.columns_info = columns_info
        return number_of_groups

    def split_fn(self, partition: pandas.DataFrame) -> 'tuple[pandas.DataFrame, ...]':
        ErrorMessage.catch_bugs_and_request_email(failure_condition=self.columns_info is None, extra_log="The 'split_fn' doesn't have proper metadata, the probable reason is that it was called before 'pivot_fn'")
        return self.split_partitions_using_pivots_for_sort(partition, self.columns_info, self.ascending, keys_are_index_levels=self.level is not None, closed_on_right=self.closed_on_right, **self.kwargs)

    @staticmethod
    def _find_quantiles(df: Union[pandas.DataFrame, pandas.Series], quantiles: list, method: str) -> np.ndarray:
        """
        Find quantiles of a given dataframe using the specified method.

        We use this method to provide backwards compatibility with NumPy versions < 1.23 (e.g. when
        the user is using Modin in compat mode). This is basically a wrapper around `np.quantile` that
        ensures we provide the correct `method` argument - i.e. if we are dealing with objects (which
        may or may not support algebra), we do not want to use a method to find quantiles that will
        involve algebra operations (e.g. mean) between the objects, since that may fail.

        Parameters
        ----------
        df : pandas.DataFrame or pandas.Series
            The data to pick quantiles from.
        quantiles : list[float]
            The quantiles to compute.
        method : str
            The method to use. `linear` if dealing with numeric types, otherwise `inverted_cdf`.

        Returns
        -------
        np.ndarray
            A NumPy array with the quantiles of the data.
        """
        if method == 'linear':
            return np.unique(np.quantile(df, quantiles))
        else:
            try:
                return np.unique(np.quantile(df, quantiles, method=method))
            except Exception:
                return np.unique(np.quantile(df, quantiles, interpolation='lower'))

    @staticmethod
    def pick_samples_for_quantiles(df: pandas.DataFrame, num_partitions: int, length: int) -> pandas.DataFrame:
        """
        Pick samples over the given partition.

        This function picks samples from the given partition using the TeraSort algorithm - each
        value is sampled with probability 1 / m * ln(n * t) where m = total_length / num_partitions,
        t = num_partitions, and n = total_length.

        Parameters
        ----------
        df : pandas.Dataframe
            The masked dataframe to pick samples from.
        num_partitions : int
            The number of partitions.
        length : int
            The total length.

        Returns
        -------
        pandas.DataFrame:
            The samples for the partition.

        Notes
        -----
        This sampling algorithm is inspired by TeraSort. You can find more information about TeraSort
        and the sampling algorithm at https://www.cse.cuhk.edu.hk/~taoyf/paper/sigmod13-mr.pdf.
        """
        m = length / num_partitions
        probability = 1 / m * np.log(num_partitions * length)
        return df.sample(frac=probability)

    def pick_pivots_from_samples_for_sort(self, samples: pandas.Series, ideal_num_new_partitions: int, method: str='linear', key: Optional[Callable]=None) -> np.ndarray:
        """
        Determine quantiles from the given samples.

        This function takes as input the quantiles calculated over all partitions from
        `sample_func` defined above, and determines a final NPartitions.get() quantiles
        to use to roughly sort the entire dataframe. It does so by collating all the samples
        and computing NPartitions.get() quantiles for the overall set.

        Parameters
        ----------
        samples : pandas.Series
            The samples computed by ``get_partition_quantiles_for_sort``.
        ideal_num_new_partitions : int
            The ideal number of new partitions.
        method : str, default: linear
            The method to use when picking quantiles.
        key : Callable, default: None
            The key to use on the samples when picking pivots.

        Returns
        -------
        np.ndarray
            A list of overall quantiles.
        """
        samples = samples.to_numpy()
        if key is not None:
            samples = key(samples)
        num_quantiles = ideal_num_new_partitions
        quantiles = [i / num_quantiles for i in range(1, num_quantiles)]
        if len(quantiles) > 0:
            return self._find_quantiles(samples, quantiles, method)
        return np.array([])

    @staticmethod
    def split_partitions_using_pivots_for_sort(df: pandas.DataFrame, columns_info: 'list[ColumnInfo]', ascending: bool, keys_are_index_levels: bool=False, closed_on_right: bool=False, **kwargs: dict) -> 'tuple[pandas.DataFrame, ...]':
        """
        Split the given dataframe into the partitions specified by `pivots` in `columns_info`.

        This function takes as input a row-axis partition, as well as the quantiles determined
        by the `pivot_func` defined above. It then splits the input dataframe into NPartitions.get()
        dataframes, with the elements in the i-th split belonging to the i-th partition, as determined
        by the quantiles we're using.

        Parameters
        ----------
        df : pandas.Dataframe
            The partition to split.
        columns_info : list of ColumnInfo
            Information regarding keys and pivots for range partitioning.
        ascending : bool
            The ascending flag.
        keys_are_index_levels : bool, default: False
            Whether `columns_info` describes index levels or actual columns from `df`.
        closed_on_right : bool, default: False
            Whether to include the right limit in range-partitioning.
                True:  bins[i - 1] < x <= bins[i]
                False: bins[i - 1] <= x < bins[i]
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[pandas.DataFrame]
            A tuple of the splits from this partition.
        """
        if len(columns_info) == 0:
            return (df,)
        key_data = ShuffleSortFunctions._index_to_df_zero_copy(df, [col_info.name for col_info in columns_info]) if keys_are_index_levels else df[[col_info.name for col_info in columns_info]]
        na_index = key_data.isna().squeeze(axis=1)
        if na_index.ndim == 2:
            na_index = na_index.any(axis=1)
        na_rows = df[na_index]
        non_na_rows = df[~na_index]

        def get_group(grp, key, df):
            """Get a group with the `key` from the `grp`, if it doesn't exist return an empty slice of `df`."""
            try:
                return grp.get_group(key)
            except KeyError:
                return pandas.DataFrame(index=df.index[:0], columns=df.columns).astype(df.dtypes)
        groupby_codes = []
        group_keys = []
        for col_info in columns_info:
            pivots = col_info.pivots
            if len(pivots) == 0:
                continue
            if not ascending and col_info.is_numeric:
                pivots = pivots[::-1]
            group_keys.append(range(len(pivots) + 1))
            key = kwargs.pop('key', None)
            cols_to_digitize = non_na_rows.index.get_level_values(col_info.name) if keys_are_index_levels else non_na_rows[col_info.name]
            if key is not None:
                cols_to_digitize = key(cols_to_digitize)
            if cols_to_digitize.ndim == 2:
                cols_to_digitize = cols_to_digitize.squeeze()
            if col_info.is_numeric:
                groupby_col = np.digitize(cols_to_digitize, pivots, right=closed_on_right)
                if not ascending and len(np.unique(pivots)) == 1:
                    groupby_col = len(pivots) - groupby_col
            else:
                groupby_col = np.searchsorted(pivots, cols_to_digitize, side='left' if closed_on_right else 'right')
                if not ascending:
                    groupby_col = len(pivots) - groupby_col
            groupby_codes.append(groupby_col)
        if len(group_keys) == 0:
            return (df,)
        elif len(group_keys) == 1:
            group_keys = group_keys[0]
        else:
            group_keys = pandas.MultiIndex.from_product(group_keys)
        if len(non_na_rows) == 1:
            groups = [pandas.DataFrame(index=df.index[:0], columns=df.columns).astype(df.dtypes) if key != groupby_codes[0] else non_na_rows for key in group_keys]
        else:
            grouped = non_na_rows.groupby(groupby_codes)
            groups = [get_group(grouped, key, df) for key in group_keys]
        index_to_insert_na_vals = -1 if kwargs.get('na_position', 'last') == 'last' else 0
        groups[index_to_insert_na_vals] = pandas.concat([groups[index_to_insert_na_vals], na_rows]).astype(df.dtypes)
        return tuple(groups)

    @staticmethod
    def _index_to_df_zero_copy(df: pandas.DataFrame, levels: list[Union[str, int]]) -> pandas.DataFrame:
        """
        Convert index `level` of `df` to a ``pandas.DataFrame``.

        Parameters
        ----------
        df : pandas.DataFrame
        levels : list of labels or ints
            Index level to convert to a dataframe.

        Returns
        -------
        pandas.DataFrame
            The columns in the resulting dataframe use the same data arrays as the index levels
            in the original `df`, so no copies.
        """
        data = {df.index.names[lvl] if isinstance(lvl, int) else lvl: df.index.get_level_values(lvl) for lvl in levels}
        index_data = pandas.DataFrame(data, index=df.index, copy=False)
        return index_data