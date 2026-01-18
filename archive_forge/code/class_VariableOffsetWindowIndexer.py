from __future__ import annotations
from datetime import timedelta
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import ensure_platform_int
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.tseries.offsets import Nano
class VariableOffsetWindowIndexer(BaseIndexer):
    """
    Calculate window boundaries based on a non-fixed offset such as a BusinessDay.

    Examples
    --------
    >>> from pandas.api.indexers import VariableOffsetWindowIndexer
    >>> df = pd.DataFrame(range(10), index=pd.date_range("2020", periods=10))
    >>> offset = pd.offsets.BDay(1)
    >>> indexer = VariableOffsetWindowIndexer(index=df.index, offset=offset)
    >>> df
                0
    2020-01-01  0
    2020-01-02  1
    2020-01-03  2
    2020-01-04  3
    2020-01-05  4
    2020-01-06  5
    2020-01-07  6
    2020-01-08  7
    2020-01-09  8
    2020-01-10  9
    >>> df.rolling(indexer).sum()
                   0
    2020-01-01   0.0
    2020-01-02   1.0
    2020-01-03   2.0
    2020-01-04   3.0
    2020-01-05   7.0
    2020-01-06  12.0
    2020-01-07   6.0
    2020-01-08   7.0
    2020-01-09   8.0
    2020-01-10   9.0
    """

    def __init__(self, index_array: np.ndarray | None=None, window_size: int=0, index: DatetimeIndex | None=None, offset: BaseOffset | None=None, **kwargs) -> None:
        super().__init__(index_array, window_size, **kwargs)
        if not isinstance(index, DatetimeIndex):
            raise ValueError('index must be a DatetimeIndex.')
        self.index = index
        if not isinstance(offset, BaseOffset):
            raise ValueError('offset must be a DateOffset-like object.')
        self.offset = offset

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values: int=0, min_periods: int | None=None, center: bool | None=None, closed: str | None=None, step: int | None=None) -> tuple[np.ndarray, np.ndarray]:
        if step is not None:
            raise NotImplementedError('step not implemented for variable offset window')
        if num_values <= 0:
            return (np.empty(0, dtype='int64'), np.empty(0, dtype='int64'))
        if closed is None:
            closed = 'right' if self.index is not None else 'both'
        right_closed = closed in ['right', 'both']
        left_closed = closed in ['left', 'both']
        if self.index[num_values - 1] < self.index[0]:
            index_growth_sign = -1
        else:
            index_growth_sign = 1
        offset_diff = index_growth_sign * self.offset
        start = np.empty(num_values, dtype='int64')
        start.fill(-1)
        end = np.empty(num_values, dtype='int64')
        end.fill(-1)
        start[0] = 0
        if right_closed:
            end[0] = 1
        else:
            end[0] = 0
        zero = timedelta(0)
        for i in range(1, num_values):
            end_bound = self.index[i]
            start_bound = end_bound - offset_diff
            if left_closed:
                start_bound -= Nano(1)
            start[i] = i
            for j in range(start[i - 1], i):
                start_diff = (self.index[j] - start_bound) * index_growth_sign
                if start_diff > zero:
                    start[i] = j
                    break
            end_diff = (self.index[end[i - 1]] - end_bound) * index_growth_sign
            if end_diff == zero and (not right_closed):
                end[i] = end[i - 1] + 1
            elif end_diff <= zero:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]
            if not right_closed:
                end[i] -= 1
        return (start, end)