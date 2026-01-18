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
@staticmethod
def _table_from_pydict(columns: Dict[str, List[Any]]) -> Block:
    for col_name, col in columns.items():
        if col_name == TENSOR_COLUMN_NAME or isinstance(next(iter(col), None), np.ndarray):
            from ray.data.extensions.tensor_extension import ArrowTensorArray
            columns[col_name] = ArrowTensorArray.from_numpy(col)
    return pyarrow.Table.from_pydict(columns)