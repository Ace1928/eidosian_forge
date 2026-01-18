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
def _is_padded_shape_compatible_with(padded_shape, input_component_shape):
    """Returns `True` if `input_component_shape` can be padded to `padded_shape`.

  Args:
    padded_shape: A `tf.TensorShape`.
    input_component_shape: A `tf.TensorShape`.

  Returns:
    `True` if `input_component_shape` can be padded to `padded_shape`, otherwise
    `False`.
  """
    if padded_shape.dims is None or input_component_shape.dims is None:
        return True
    if len(padded_shape.dims) != len(input_component_shape.dims):
        return False
    for padded_dim, input_dim in zip(padded_shape.dims, input_component_shape.dims):
        if padded_dim.value is not None and input_dim.value is not None and (padded_dim.value < input_dim.value):
            return False
    return True