from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
class LeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit.

  It allows a small gradient when the unit is not active:

  ```
    f(x) = alpha * x if x < 0
    f(x) = x if x >= 0
  ```

  Usage:

  >>> layer = tf.keras.layers.LeakyReLU()
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-0.9, -0.3, 0.0, 2.0]
  >>> layer = tf.keras.layers.LeakyReLU(alpha=0.1)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-0.3, -0.1, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Args:
    alpha: Float >= 0. Negative slope coefficient. Default to 0.3.

  """

    def __init__(self, alpha=0.3, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        if alpha is None:
            raise ValueError('The alpha value of a Leaky ReLU layer cannot be None, needs a float. Got %s' % alpha)
        self.supports_masking = True
        self.alpha = backend.cast_to_floatx(alpha)

    def call(self, inputs):
        return backend.relu(inputs, alpha=self.alpha)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(LeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape