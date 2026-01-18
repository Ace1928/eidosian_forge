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
class Permute(Layer):
    """Permutes the dimensions of the input according to a given pattern.

  Useful e.g. connecting RNNs and convnets.

  Example:

  ```python
  model = Sequential()
  model.add(Permute((2, 1), input_shape=(10, 64)))
  # now: model.output_shape == (None, 64, 10)
  # note: `None` is the batch dimension
  ```

  Args:
    dims: Tuple of integers. Permutation pattern does not include the
      samples dimension. Indexing starts at 1.
      For instance, `(2, 1)` permutes the first and second dimensions
      of the input.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same as the input shape, but with the dimensions re-ordered according
    to the specified pattern.
  """

    def __init__(self, dims, **kwargs):
        super(Permute, self).__init__(**kwargs)
        self.dims = tuple(dims)
        if sorted(dims) != list(range(1, len(dims) + 1)):
            raise ValueError('Invalid permutation `dims` for Permute Layer: %s. The set of indices in `dims` must be consecutive and start from 1.' % (dims,))
        self.input_spec = InputSpec(ndim=len(self.dims) + 1)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = input_shape[dim]
            output_shape[i + 1] = target_dim
        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        return array_ops.transpose(inputs, perm=(0,) + self.dims)

    def get_config(self):
        config = {'dims': self.dims}
        base_config = super(Permute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))