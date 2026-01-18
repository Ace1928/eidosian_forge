import functools
import operator
import typing
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@classmethod
def _convert_values_and_partition(cls, values, row_partition, name):
    """Converts `values` and `partition` to Tensors.

    If `values` is a `RaggedTensor`, then converts `values` and `partition`
    to have compatible row-partitioning dtypes.  In particular, if any of the
    row partitioning tensors are `int64`, then all of the other row
    partitioning tensors wil be cast to `int64` (if auto_cast_partition_dtype()
    is true) or an error will be raised (if auto_cast_partition_dtype() is
    false).

    Args:
      values: The `values` for the `RaggedTensor` being constructed.
      row_partition: A RowPartition object for the `RaggedTensor` being
        constructed.
      name: The name of the RowPartition object.

    Returns:
      A tuple (values, partition).
    """
    if not isinstance(row_partition, RowPartition):
        raise TypeError(f'Argument `row_partition` must be a RowPartition. Received {row_partition}.')
    if isinstance(values, RaggedTensor):
        if values._row_partition.dtype != row_partition.dtype:
            if not ragged_config.auto_cast_partition_dtype():
                raise ValueError(f'Argument `row_partition` of RaggedTensor with name: {name} must have same dtype as Argument `values`. ({row_partition.dtype} vs. {values._row_partition.dtype}).')
            values = values.with_row_splits_dtype(row_partition.dtype)
    else:
        values = _convert_to_ragged_tensor_values(values)
    return (values, row_partition)