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
def _assert_sparse_indices_are_ragged_right(indices):
    """Checks that the given SparseTensor.indices tensor is ragged-right.

  Example: `indices = [[0, 0], [0, 1], [2, 0], [3, 1]]` is not ragged right
  because the entry `[3, 1]` skips a cell.

  Args:
    indices: The SparseTensor indices to check.

  Returns:
    A list of control dependency op tensors.
  """
    index_prefix = indices[:, :-1]
    index_suffix = indices[:, -1]
    index_prefix_changed = math_ops.reduce_any(math_ops.not_equal(index_prefix[1:], index_prefix[:-1]), axis=1)
    index_ok = array_ops.where(index_prefix_changed, math_ops.equal(index_suffix[1:], 0), math_ops.equal(index_suffix[1:], index_suffix[:-1] + 1))
    sparse_indices_are_ragged_right = math_ops.logical_and(math_ops.reduce_all(math_ops.equal(index_suffix[:1], 0)), math_ops.reduce_all(index_ok))
    message = ['SparseTensor is not right-ragged', 'SparseTensor.indices =', indices]
    return [control_flow_assert.Assert(sparse_indices_are_ragged_right, message)]