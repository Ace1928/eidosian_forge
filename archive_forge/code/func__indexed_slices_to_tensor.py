import collections
import warnings
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _indexed_slices_to_tensor(value, dtype=None, name=None, as_ref=False):
    """Converts an IndexedSlices object `value` to a Tensor.

  NOTE(mrry): This function is potentially expensive.

  Args:
    value: An ops.IndexedSlices object.
    dtype: The dtype of the Tensor to be returned.
    name: Optional name to use for the returned Tensor.
    as_ref: True if a ref is requested.

  Returns:
    A dense Tensor representing the values in the given IndexedSlices.

  Raises:
    ValueError: If the IndexedSlices does not have the same dtype.
  """
    _ = as_ref
    if dtype and (not dtype.is_compatible_with(value.dtype)):
        raise ValueError(f'Incompatible tensor conversion requested to `dtype` {dtype.name} for IndexedSlices ({value}) with dtype {value.dtype.name}')
    if value.dense_shape is None:
        raise ValueError(f'Tensor conversion requested for IndexedSlices for argument `value` without dense_shape: {value!s}')
    if not context.executing_eagerly():
        dense_shape_value = tensor_util.constant_value(value.dense_shape)
        if dense_shape_value is not None:
            num_elements = np.prod(dense_shape_value)
            if num_elements >= _LARGE_SPARSE_NUM_ELEMENTS:
                warnings.warn('Converting sparse IndexedSlices to a dense Tensor with %d elements. This may consume a large amount of memory.' % num_elements)
    return gen_math_ops.unsorted_segment_sum(value.values, value.indices, value.dense_shape[0], name=name)