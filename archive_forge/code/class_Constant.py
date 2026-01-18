import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
class Constant(Initializer):
    """Initializer that generates tensors with constant values.

  Also available via the shortcut function `tf.keras.initializers.constant`.

  Only scalar values are allowed.
  The constant value provided must be convertible to the dtype requested
  when calling the initializer.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Constant(3.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Constant(3.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    value: A Python scalar.
  """

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to `self.value`.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not specified,
       `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`).
      **kwargs: Additional keyword arguments.
    """
        del kwargs
        return constant_op.constant(self.value, dtype=_get_dtype(dtype), shape=shape)

    def get_config(self):
        return {'value': self.value}