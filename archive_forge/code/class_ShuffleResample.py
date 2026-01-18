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
@_inherit_docstrings(ShuffleSortFunctions)
class ShuffleResample(ShuffleSortFunctions):

    def __init__(self, modin_frame: 'PandasDataframe', columns: Union[str, list], ascending: Union[list, bool], ideal_num_new_partitions: int, resample_kwargs: dict, **kwargs: dict):
        resample_kwargs = resample_kwargs.copy()
        rule = resample_kwargs.pop('rule')
        if resample_kwargs['closed'] is None:
            if rule in ('ME', 'YE', 'QE', 'BME', 'BA', 'BQE', 'W'):
                resample_kwargs['closed'] = 'right'
            else:
                resample_kwargs['closed'] = 'left'
        super().__init__(modin_frame, columns, ascending, ideal_num_new_partitions, closed_on_right=resample_kwargs['closed'] == 'right', **kwargs)
        resample_kwargs['freq'] = to_offset(rule)
        self.resample_kwargs = resample_kwargs

    @staticmethod
    def pick_samples_for_quantiles(df: pandas.DataFrame, num_partitions: int, length: int) -> pandas.DataFrame:
        return pandas.concat([df.min().to_frame().T, df.max().to_frame().T])

    def pick_pivots_from_samples_for_sort(self, samples: np.ndarray, ideal_num_new_partitions: int, method: str='linear', key: Optional[Callable]=None) -> np.ndarray:
        if key is not None:
            raise NotImplementedError(key)
        max_value = samples.max()
        first, last = _get_timestamp_range_edges(samples.min(), max_value, self.resample_kwargs['freq'], unit=samples.dt.unit, closed=self.resample_kwargs['closed'], origin=self.resample_kwargs['origin'], offset=self.resample_kwargs['offset'])
        all_bins = pandas.date_range(start=first, end=last, freq=self.resample_kwargs['freq'], ambiguous=True, nonexistent='shift_forward', unit=samples.dt.unit)
        all_bins = self._adjust_bin_edges(all_bins, max_value, freq=self.resample_kwargs['freq'], closed=self.resample_kwargs['closed'])
        step = 1 / ideal_num_new_partitions
        bins = [all_bins[int(len(all_bins) * i * step)] for i in range(1, ideal_num_new_partitions)]
        return bins

    def _adjust_bin_edges(self, binner: pandas.DatetimeIndex, end_timestamp, freq, closed) -> pandas.DatetimeIndex:
        """
        Adjust bin edges.

        This function was copied & simplified from ``pandas.core.resample.TimeGrouper._adjuct_bin_edges()``.

        Parameters
        ----------
        binner : pandas.DatetimeIndex
        end_timestamp : pandas.Timestamp
        freq : str
        closed : bool

        Returns
        -------
        pandas.DatetimeIndex
        """
        if freq.name not in ('BME', 'ME', 'W') and freq.name.split('-')[0] not in ('BQE', 'BYE', 'QE', 'YE', 'W'):
            return binner
        if closed == 'right':
            edges_dti = binner.tz_localize(None)
            edges_dti = edges_dti + pandas.Timedelta(days=1, unit=edges_dti.unit).as_unit(edges_dti.unit) - pandas.Timedelta(1, unit=edges_dti.unit).as_unit(edges_dti.unit)
            binner = edges_dti.tz_localize(binner.tz)
        if binner[-2] > end_timestamp:
            binner = binner[:-1]
        return binner

    @staticmethod
    def split_partitions_using_pivots_for_sort(df: pandas.DataFrame, columns_info: 'list[ColumnInfo]', ascending: bool, closed_on_right: bool=True, **kwargs: dict) -> 'tuple[pandas.DataFrame, ...]':

        def add_attr(df, timestamp):
            if 'bin_bounds' in df.attrs:
                df.attrs['bin_bounds'] = (*df.attrs['bin_bounds'], timestamp)
            else:
                df.attrs['bin_bounds'] = (timestamp,)
            return df
        result = ShuffleSortFunctions.split_partitions_using_pivots_for_sort(df, columns_info, ascending, **kwargs)
        for i, pivot in enumerate(columns_info[0].pivots):
            add_attr(result[i], pivot - pandas.Timedelta(1, unit='ns'))
            if i + 1 <= len(result):
                add_attr(result[i + 1], pivot + pandas.Timedelta(1, unit='ns'))
        return result