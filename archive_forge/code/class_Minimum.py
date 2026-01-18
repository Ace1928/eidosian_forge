from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
class Minimum(_Merge):
    """Layer that computes the minimum (element-wise) a list of inputs.

  It takes as input a list of tensors, all of the same shape, and returns
  a single tensor (also of the same shape).

  >>> tf.keras.layers.Minimum()([np.arange(5).reshape(5, 1),
  ...                            np.arange(5, 10).reshape(5, 1)])
  <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[0],
       [1],
       [2],
       [3],
       [4]])>

  >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
  >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
  >>> minned = tf.keras.layers.Minimum()([x1, x2])
  >>> minned.shape
  TensorShape([5, 8])
  """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = math_ops.minimum(output, inputs[i])
        return output