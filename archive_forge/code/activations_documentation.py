from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import dispatch
Returns function.

  Args:
      identifier: Function or string

  Returns:
      Function corresponding to the input string or input function.

  For example:

  >>> tf.keras.activations.get('softmax')
   <function softmax at 0x1222a3d90>
  >>> tf.keras.activations.get(tf.keras.activations.softmax)
   <function softmax at 0x1222a3d90>
  >>> tf.keras.activations.get(None)
   <function linear at 0x1239596a8>
  >>> tf.keras.activations.get(abs)
   <built-in function abs>
  >>> tf.keras.activations.get('abcd')
  Traceback (most recent call last):
  ...
  ValueError: Unknown activation function:abcd

  Raises:
      ValueError: Input is an unknown function or string, i.e., the input does
      not denote any defined function.
  