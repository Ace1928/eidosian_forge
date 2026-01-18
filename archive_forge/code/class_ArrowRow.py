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
class ArrowRow(TableRow):
    """
    Row of a tabular Dataset backed by a Arrow Table block.
    """

    def __getitem__(self, key: Union[str, List[str]]) -> Any:
        from ray.data.extensions.tensor_extension import ArrowTensorType, ArrowVariableShapedTensorType

        def get_item(keys: List[str]) -> Any:
            schema = self._row.schema
            if isinstance(schema.field(keys[0]).type, (ArrowTensorType, ArrowVariableShapedTensorType)):
                return tuple([ArrowBlockAccessor._build_tensor_row(self._row, col_name=key) for key in keys])
            table = self._row.select(keys)
            if len(table) == 0:
                return None
            items = [col[0] for col in table.columns]
            try:
                return tuple([item.as_py() for item in items])
            except AttributeError:
                return items
        is_single_item = isinstance(key, str)
        keys = [key] if is_single_item else key
        items = get_item(keys)
        if items is None:
            return None
        elif is_single_item:
            return items[0]
        else:
            return items

    def __iter__(self) -> Iterator:
        for k in self._row.column_names:
            yield k

    def __len__(self):
        return self._row.num_columns