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
class SlicingOpLambda(TFOpLambda):
    """Wraps TF API symbols in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF symbol on KerasTensors.

  Like Lambda layers, this layer tries to raise warnings when it detects users
  explicitly use variables in the call. (To let them know
  that the layer will not capture the variables).

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = tf.Variable(...)
  out = x * tf_variable
  """

    @trackable.no_automatic_dependency_tracking
    def __init__(self, function, **kwargs):
        super(SlicingOpLambda, self).__init__(function, **kwargs)
        original_call = self.call

        def _call_wrapper(*args, **kwargs):
            new_args = []
            for arg in args:
                arg = _dict_to_slice(arg)
                if isinstance(arg, (list, tuple)):
                    new_arg = []
                    for sub_arg in arg:
                        new_arg.append(_dict_to_slice(sub_arg))
                    arg = new_arg
                new_args.append(arg)
            new_kwargs = {}
            for key, value in kwargs.items():
                value = _dict_to_slice(value)
                if isinstance(value, (list, tuple)):
                    new_value = []
                    for v in value:
                        new_value.append(_dict_to_slice(v))
                    value = new_value
                new_kwargs[key] = value
            return original_call(*new_args, **new_kwargs)
        self.call = tf_decorator.make_decorator(original_call, _call_wrapper)