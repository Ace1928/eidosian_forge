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
class InstanceProperty(Layer):
    """Wraps an instance property access (e.g. `x.foo`) in a Keras Layer.

  This layer takes an attribute name `attr_name` in the constructor and,
  when called on input tensor `obj` returns `obj.attr_name`.

  KerasTensors specialized for specific extension types use it to
  represent instance property accesses on the represented object in the
  case where the property needs to be dynamically accessed as opposed to
  being statically computed from the typespec, e.g.

  x = keras.Input(..., ragged=True)
  out = x.flat_values
  """

    @trackable.no_automatic_dependency_tracking
    def __init__(self, attr_name, **kwargs):
        self.attr_name = attr_name
        if 'name' not in kwargs:
            kwargs['name'] = K.unique_object_name('input.' + self.attr_name, zero_based=True, avoid_observed_names=True)
        kwargs['autocast'] = False
        self._must_restore_from_config = True
        super(InstanceProperty, self).__init__(**kwargs)
        self._preserve_input_structure_in_config = True

    def call(self, obj):
        return getattr(obj, self.attr_name)

    def get_config(self):
        config = {'attr_name': self.attr_name}
        base_config = super(InstanceProperty, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)