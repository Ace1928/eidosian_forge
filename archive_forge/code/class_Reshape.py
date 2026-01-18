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
class Reshape(Layer):
    """Layer that reshapes inputs into the given shape.

  Input shape:
    Arbitrary, although all dimensions in the input shape must be known/fixed.
    Use the keyword argument `input_shape` (tuple of integers, does not include
    the samples/batch size axis) when using this layer as the first layer
    in a model.

  Output shape:
    `(batch_size,) + target_shape`

  Example:

  >>> # as first layer in a Sequential model
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Reshape((3, 4), input_shape=(12,)))
  >>> # model.output_shape == (None, 3, 4), `None` is the batch size.
  >>> model.output_shape
  (None, 3, 4)

  >>> # as intermediate layer in a Sequential model
  >>> model.add(tf.keras.layers.Reshape((6, 2)))
  >>> model.output_shape
  (None, 6, 2)

  >>> # also supports shape inference using `-1` as dimension
  >>> model.add(tf.keras.layers.Reshape((-1, 2, 2)))
  >>> model.output_shape
  (None, 3, 2, 2)
  """

    def __init__(self, target_shape, **kwargs):
        """Creates a `tf.keras.layers.Reshape`  layer instance.

    Args:
      target_shape: Target shape. Tuple of integers, does not include the
        samples dimension (batch size).
      **kwargs: Any additional layer keyword arguments.
    """
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Find and replace a missing dimension in an output shape.

    This is a near direct port of the internal Numpy function
    `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

    Args:
      input_shape: Shape of array being reshaped
      output_shape: Desired shape of the array with at most
        a single -1 which indicates a dimension that should be
        derived from the input shape.

    Returns:
      The new output shape with a -1 replaced with its computed value.

    Raises:
      ValueError: If the total array size of the output_shape is
      different than the input_shape, or more than one unknown dimension
      is specified.
    """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged, input_shape = {}, output_shape = {}'.format(input_shape, output_shape)
        known, unknown = (1, None)
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim
        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)
        return output_shape

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if None in input_shape[1:]:
            output_shape = [input_shape[0]]
            output_shape += tuple((s if s != -1 else None for s in self.target_shape))
        else:
            output_shape = [input_shape[0]]
            output_shape += self._fix_unknown_dimension(input_shape[1:], self.target_shape)
        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        result = array_ops.reshape(inputs, (array_ops.shape(inputs)[0],) + self.target_shape)
        if not context.executing_eagerly():
            result.set_shape(self.compute_output_shape(inputs.shape))
        return result

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))