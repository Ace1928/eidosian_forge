from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
Returns a `tf.Tensor` that represents the given shape.

  Args:
    shape_like: A value that can be converted to a `tf.TensorShape` or a
      `tf.Tensor`.

  Returns:
    A 1-D `tf.Tensor` of `tf.int64` elements representing the given shape, where
    `-1` is substituted for any unknown dimensions.
  