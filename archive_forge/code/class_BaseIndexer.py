from __future__ import annotations
from datetime import timedelta
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import ensure_platform_int
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.tseries.offsets import Nano
class BaseIndexer:
    """
    Base class for window bounds calculations.

    Examples
    --------
    >>> from pandas.api.indexers import BaseIndexer
    >>> class CustomIndexer(BaseIndexer):
    ...     def get_window_bounds(self, num_values, min_periods, center, closed, step):
    ...         start = np.empty(num_values, dtype=np.int64)
    ...         end = np.empty(num_values, dtype=np.int64)
    ...         for i in range(num_values):
    ...             start[i] = i
    ...             end[i] = i + self.window_size
    ...         return start, end
    >>> df = pd.DataFrame({"values": range(5)})
    >>> indexer = CustomIndexer(window_size=2)
    >>> df.rolling(indexer).sum()
        values
    0	1.0
    1	3.0
    2	5.0
    3	7.0
    4	4.0
    """

    def __init__(self, index_array: np.ndarray | None=None, window_size: int=0, **kwargs) -> None:
        self.index_array = index_array
        self.window_size = window_size
        for key, value in kwargs.items():
            setattr(self, key, value)

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values: int=0, min_periods: int | None=None, center: bool | None=None, closed: str | None=None, step: int | None=None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError