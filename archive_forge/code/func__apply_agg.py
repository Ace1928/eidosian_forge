import collections
import heapq
from typing import (
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import find_partitions
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.row import TableRow
def _apply_agg(self, agg_fn: Callable[['pandas.Series', bool], U], on: str) -> Optional[U]:
    """Helper providing null handling around applying an aggregation to a column."""
    pd = lazy_import_pandas()
    if on is not None and (not isinstance(on, str)):
        raise ValueError(f'on must be a string or None when aggregating on Pandas blocks, but got: {type(on)}.')
    if self.num_rows() == 0:
        return None
    col = self._table[on]
    try:
        val = agg_fn(col)
    except TypeError as e:
        if np.issubdtype(col.dtype, np.object_) and col.isnull().all():
            return None
        raise e from None
    if pd.isnull(val):
        return None
    return val