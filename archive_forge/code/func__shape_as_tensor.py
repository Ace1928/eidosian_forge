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
def _shape_as_tensor(shape, dtype):
    """Takes shape and coerces it to a shape as a tensor.

  If the object is already a tensor, simply passes it on (result is guaranteed
  to be int64 or int32, but not necessarily dtype).
  If not, creates a tensor of type dtype.

  Result is either a scalar equal to -1 if the shape is unknown_rank.
  Otherwise, it is a vector, where unknown dimensions are represented with a
  value of -1.

  In C++, see TensorShapeFromTensor for parsing shapes in kernels, and
  InferenceContext::MakeShapeFromShapeTensorTreatScalarAsUnknownShape, for
  use in the shape inference function.

  Args:
    shape: input to coerce from TensorShape, Tensor, None, List[Optional[Int]],
      Tuple[Optional[Int]].
    dtype: tf.int64 or tf.int32

  Returns:
    a scalar or vector tensor of dtype tf.int32 or tf.int64.
  """
    if dtype != dtypes.int64 and dtype != dtypes.int32:
        raise ValueError(f'Expected int64 or int32 for dtype: got {dtype}.')
    if isinstance(shape, tensor_lib.Tensor):
        if shape.dtype != dtypes.int64 and shape.dtype != dtypes.int32:
            return math_ops.cast(shape, dtype)
        return shape
    shape = tensor_shape.as_shape(shape)
    if not shape:
        return constant_op.constant(-1, dtype=dtype)
    shape = [-1 if x is None else x for x in shape.as_list()]
    return constant_op.constant(shape, dtype=dtype)