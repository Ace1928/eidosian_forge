import collections
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, TypeVar, Union
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.block_builder import BlockBuilder
from ray.data._internal.numpy_support import convert_udf_returns_to_numpy, is_array_like
from ray.data._internal.size_estimator import SizeEstimator
from ray.data.block import Block, BlockAccessor
from ray.data.row import TableRow
def _compact_if_needed(self) -> None:
    assert self._columns
    if self._uncompacted_size.size_bytes() < MAX_UNCOMPACTED_SIZE_BYTES:
        return
    columns = {key: convert_udf_returns_to_numpy(col) for key, col in self._columns.items()}
    block = self._table_from_pydict(columns)
    self.add_block(block)
    self._uncompacted_size = SizeEstimator()
    self._columns.clear()
    self._num_compactions += 1