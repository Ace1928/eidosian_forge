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
def _build_tensor_row(row: ArrowRow, col_name: str=TENSOR_COLUMN_NAME) -> np.ndarray:
    from pkg_resources._vendor.packaging.version import parse as parse_version
    element = row[col_name][0]
    pyarrow_version = _get_pyarrow_version()
    if pyarrow_version is not None:
        pyarrow_version = parse_version(pyarrow_version)
    if pyarrow_version is None or pyarrow_version >= parse_version('8.0.0'):
        assert isinstance(element, pyarrow.ExtensionScalar)
        if pyarrow_version is None or pyarrow_version >= parse_version('9.0.0'):
            element = element.as_py()
        else:
            element = element.type._extension_scalar_to_ndarray(element)
    assert isinstance(element, np.ndarray), type(element)
    return element