from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
Whether tensor `t` supports creating a default gradient.

  This function assumes that `t` is of a trainable type.

  Args:
    t: Tensor

  Returns:
    Bool
  