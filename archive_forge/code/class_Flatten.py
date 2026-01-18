import warnings
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.legacy_tf_layers import base
from tensorflow.python.ops import init_ops
class Flatten(keras_layers.Flatten, base.Layer):
    """Flattens an input tensor while preserving the batch axis (axis 0).

  Args:
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.

  Examples:

  ```
    x = tf.compat.v1.placeholder(shape=(None, 4, 4), dtype='float32')
    y = Flatten()(x)
    # now `y` has shape `(None, 16)`

    x = tf.compat.v1.placeholder(shape=(None, 3, None), dtype='float32')
    y = Flatten()(x)
    # now `y` has shape `(None, None)`
  ```
  """
    pass