import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _merge_row_partitions(row_partitions, value, rank, dtype, validate):
    """Merges `row_partitions` with `row_partitions(value)`."""
    if isinstance(value, tensor.Tensor):
        value_row_partitions = _row_partitions_for_tensor(value, rank, dtype)
    elif isinstance(value, ragged_tensor.RaggedTensor):
        value_row_partitions = _row_partitions_for_ragged_tensor(value, rank, dtype)
    else:
        assert isinstance(value, StructuredTensor), type(value)
        value_row_partitions = value.row_partitions[:rank - 1]
    assert len(value_row_partitions) == rank - 1
    if row_partitions is None:
        return tuple(value_row_partitions)
    else:
        return tuple([p1._merge_precomputed_encodings(p2, validate) for p1, p2 in zip(row_partitions, value_row_partitions)])