import copy
import functools
import operator
import sys
import textwrap
import types as python_types
import warnings
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name
class Activation(Layer):
    """Applies an activation function to an output.

  Args:
    activation: Activation function, such as `tf.nn.relu`, or string name of
      built-in activation function, such as "relu".

  Usage:

  >>> layer = tf.keras.layers.Activation('relu')
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.Activation(tf.nn.relu)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)

    def call(self, inputs):
        return self.activation(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))