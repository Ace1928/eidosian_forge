import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@classmethod
def _from_spec(cls, spec: Union['DynamicRaggedShape.Spec', ragged_tensor.RaggedTensorSpec, tensor_lib.TensorSpec], dtype: dtypes.DType=dtypes.int64) -> 'DynamicRaggedShape.Spec':
    """Create a TypeSpec for the shape of an object with a given TypeSpec.

      I.e., if `x_spec = tf.type_spec_from_value(x)`, then
      `DynamicRaggedShape.from_spec(x_spec)` returns a TypeSpec compatible with
      `tf.type_spec_from_value(tf.shape(x))`.

      >>> rt = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
      >>> rt_spec = tf.type_spec_from_value(rt)
      >>> rt_shape = DynamicRaggedShape.from_tensor(rt)

      >>> shape_spec_1 = tf.type_spec_from_value(rt_shape)
      >>> shape_spec_2 = DynamicRaggedShape.Spec._from_spec(rt_spec)
      >>> assert shape_spec_1.is_compatible_with(shape_spec_2)

      Args:
        spec: a Spec of a Tensor or RaggedTensor.
        dtype: the default dtype (if necessary).

      Returns:
        A Spec of the shape of a Tensor or RaggedTensor.

      """
    if isinstance(spec, DynamicRaggedShape.Spec):
        return spec
    elif isinstance(spec, ragged_tensor.RaggedTensorSpec):
        return cls._from_tensor_shape(spec.shape, spec.ragged_rank, spec.row_splits_dtype)
    elif isinstance(spec, tensor_lib.TensorSpec):
        return cls._from_tensor_shape(shape=spec.shape, num_row_partitions=0, dtype=dtype)