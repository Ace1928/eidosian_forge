from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
class ReLU(Layer):
    """Rectified Linear Unit activation function.

  With default values, it returns element-wise `max(x, 0)`.

  Otherwise, it follows:

  ```
    f(x) = max_value if x >= max_value
    f(x) = x if threshold <= x < max_value
    f(x) = negative_slope * (x - threshold) otherwise
  ```

  Usage:

  >>> layer = tf.keras.layers.ReLU()
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.ReLU(max_value=1.0)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 1.0]
  >>> layer = tf.keras.layers.ReLU(negative_slope=1.0)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-3.0, -1.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.ReLU(threshold=1.5)
  >>> output = layer([-3.0, -1.0, 1.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Args:
    max_value: Float >= 0. Maximum activation value. Default to None, which
      means unlimited.
    negative_slope: Float >= 0. Negative slope coefficient. Default to 0.
    threshold: Float >= 0. Threshold value for thresholded activation. Default
      to 0.
  """

    def __init__(self, max_value=None, negative_slope=0, threshold=0, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        if max_value is not None and max_value < 0.0:
            raise ValueError('max_value of a ReLU layer cannot be a negative value. Got: %s' % max_value)
        if negative_slope is None or negative_slope < 0.0:
            raise ValueError('negative_slope of a ReLU layer cannot be a negative value. Got: %s' % negative_slope)
        if threshold is None or threshold < 0.0:
            raise ValueError('threshold of a ReLU layer cannot be a negative value. Got: %s' % threshold)
        self.supports_masking = True
        if max_value is not None:
            max_value = backend.cast_to_floatx(max_value)
        self.max_value = max_value
        self.negative_slope = backend.cast_to_floatx(negative_slope)
        self.threshold = backend.cast_to_floatx(threshold)

    def call(self, inputs):
        return backend.relu(inputs, alpha=self.negative_slope, max_value=self.max_value, threshold=self.threshold)

    def get_config(self):
        config = {'max_value': self.max_value, 'negative_slope': self.negative_slope, 'threshold': self.threshold}
        base_config = super(ReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape