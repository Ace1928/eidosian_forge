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
class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

  Also available via the shortcut function `tf.keras.initializers.zeros`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Zeros()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Zeros()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
  """

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
       supported. If not specified, `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`).
      **kwargs: Additional keyword arguments.
    """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if not dtype.is_numpy_compatible or dtype == dtypes.string:
            raise ValueError('Expected numeric or boolean dtype, got %s.' % dtype)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return array_ops.zeros(shape, dtype)