from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasUtils
from qpd import QPDEngine, run_sql
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
class _RowsIndexer(BaseIndexer):

    def __init__(self, start: Optional[int], end: Optional[int], **kwargs: Any):
        window = 0 if start is None or end is None else end - start + 1
        kw = dict(kwargs)
        kw['window_size'] = window
        kw['start'] = start
        kw['end'] = end
        super().__init__(**kw)

    def get_window_bounds(self, num_values: int=0, min_periods: Optional[int]=None, center: Optional[bool]=None, closed: Optional[str]=None, step: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        if self.start is None:
            start = np.zeros(num_values, dtype=np.int64)
        else:
            start = np.arange(self.start, self.start + num_values, dtype=np.int64)
            if self.start < 0:
                start[:-self.start] = 0
            elif self.start > 0:
                start[-self.start:] = num_values
        if self.end is None:
            end = np.full(num_values, num_values, dtype=np.int64)
        else:
            end = np.arange(self.end + 1, self.end + 1 + num_values, dtype=np.int64)
            if self.end > 0:
                end[-self.end:] = num_values
            elif self.end < 0:
                end[:-self.end] = 0
        return (start, end)