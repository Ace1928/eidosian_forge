import collections
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, TypeVar, Union
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.block_builder import BlockBuilder
from ray.data._internal.numpy_support import convert_udf_returns_to_numpy, is_array_like
from ray.data._internal.size_estimator import SizeEstimator
from ray.data.block import Block, BlockAccessor
from ray.data.row import TableRow
def _get_row(self, index: int, copy: bool=False) -> Union[TableRow, np.ndarray]:
    base_row = self.slice(index, index + 1, copy=copy)
    row = self.ROW_TYPE(base_row)
    return row