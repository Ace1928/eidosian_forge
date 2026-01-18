from __future__ import annotations
from datetime import timedelta
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import ensure_platform_int
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.tseries.offsets import Nano
class FixedWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of fixed length."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values: int=0, min_periods: int | None=None, center: bool | None=None, closed: str | None=None, step: int | None=None) -> tuple[np.ndarray, np.ndarray]:
        if center or self.window_size == 0:
            offset = (self.window_size - 1) // 2
        else:
            offset = 0
        end = np.arange(1 + offset, num_values + 1 + offset, step, dtype='int64')
        start = end - self.window_size
        if closed in ['left', 'both']:
            start -= 1
        if closed in ['left', 'neither']:
            end -= 1
        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)
        return (start, end)