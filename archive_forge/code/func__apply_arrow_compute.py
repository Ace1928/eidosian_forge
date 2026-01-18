import collections
import heapq
import random
from typing import (
import numpy as np
from ray._private.utils import _get_pyarrow_version
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.arrow_ops import transform_polars, transform_pyarrow
from ray.data._internal.numpy_support import (
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import _truncated_repr, find_partitions
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.row import TableRow
def _apply_arrow_compute(self, compute_fn: Callable, on: str, ignore_nulls: bool) -> Optional[U]:
    """Helper providing null handling around applying an aggregation to a column."""
    import pyarrow as pa
    if not isinstance(on, str):
        raise ValueError(f'on must be a string when aggregating on Arrow blocks, but got:{type(on)}.')
    if self.num_rows() == 0:
        return None
    col = self._table[on]
    if pa.types.is_null(col.type):
        return None
    else:
        return compute_fn(col, skip_nulls=ignore_nulls).as_py()