import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
def _padded_shape_to_tensor(padded_shape, input_component_shape):
    """Converts `padded_shape` to a `tf.Tensor` representing that shape.

  Args:
    padded_shape: A shape-like object, which may be a `tf.TensorShape`, a Python
      sequence, or a 1-D `tf.Tensor` of `tf.int64` elements.
    input_component_shape: A `tf.TensorShape`, with which `padded_shape` must be
      compatible.

  Returns:
    A 1-D `tf.Tensor` of `tf.int64` elements, representing `padded_shape`.

  Raises:
    ValueError: If `padded_shape` is not a shape or not compatible with
      `input_component_shape`.
    TypeError: If `padded_shape` is not convertible to a `tf.int64` tensor.
  """
    try:
        padded_shape_as_shape = tensor_shape.as_shape(padded_shape)
        ret = ops.convert_to_tensor([dim if dim is not None else -1 for dim in padded_shape_as_shape.as_list()], dtype=dtypes.int64)
    except (TypeError, ValueError) as e:
        ret = ops.convert_to_tensor(padded_shape, preferred_dtype=dtypes.int64)
        if ret.shape.dims is not None and len(ret.shape.dims) != 1:
            raise ValueError(f'Padded shape {padded_shape} must be a `tf.int64` vector tensor, but its shape was {ret.shape}.') from e
        if ret.dtype != dtypes.int64:
            raise TypeError(f'Padded shape {padded_shape} must be a `tf.int64` vector tensor, but its element type was {ret.dtype.name}.') from e
        padded_shape_as_shape = tensor_util.constant_value_as_shape(ret)
    if not _is_padded_shape_compatible_with(padded_shape_as_shape, input_component_shape):
        raise ValueError(f'The padded shape {padded_shape_as_shape} is not compatible with the shape {input_component_shape} of the corresponding input component.')
    return ret