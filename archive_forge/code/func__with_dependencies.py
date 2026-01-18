import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _with_dependencies(self, dependencies):
    """Returns a new RowPartition equal to self with control dependencies.

    Specifically, self._row_splits is gated by the given control dependencies.
    Used to add sanity checks to the constructors.

    Args:
      dependencies: a list of tensors to use as dependencies.

    Returns:
      A new RowPartition object.
    """
    new_row_splits = control_flow_ops.with_dependencies(dependencies, self._row_splits)
    return RowPartition(row_splits=new_row_splits, row_lengths=self._row_lengths, value_rowids=self._value_rowids, nrows=self._nrows, uniform_row_length=self._uniform_row_length, internal=_row_partition_factory_key)