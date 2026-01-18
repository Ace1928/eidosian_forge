import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
class SimpleIndexer(BaseIndexer):

    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None, step=None):
        min_periods = self.window_size if min_periods is None else 0
        end = np.arange(num_values, dtype=np.int64) + 1
        start = end.copy() - self.window_size
        start[start < 0] = min_periods
        return (start, end)