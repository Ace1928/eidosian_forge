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
def _merge_inner_shape(inner_shape: tensor_lib.Tensor, static_inner_shape: tensor_shape.TensorShape, outer_axis: int, inner_axis: int) -> Tuple[tensor_lib.Tensor, tensor_shape.TensorShape]:
    """Merge the inner shape of a DynamicRaggedShape."""
    prefix = inner_shape[:outer_axis]
    suffix = inner_shape[inner_axis + 1:]
    internal = inner_shape[outer_axis:inner_axis + 1]
    internal_value = [_reduce_prod_patch(internal)]
    new_internal = array_ops.concat([prefix, internal_value, suffix], axis=0)
    prefix_static = static_inner_shape[:outer_axis]
    suffix_static = static_inner_shape[inner_axis + 1:]
    internal_static = static_inner_shape[outer_axis:inner_axis + 1]
    internal_value_static = tensor_shape.TensorShape([internal_static.num_elements()])
    new_internal_static = prefix_static + internal_value_static + suffix_static
    return (new_internal, new_internal_static)