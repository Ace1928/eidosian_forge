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
class ClassMethod(Layer):
    """Wraps a TF API Class's class method  in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF Class's class method on KerasTensors.

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = keras.Input(...)
  out = tf.RaggedTensor.from_row_splits(x, y)
  """

    @trackable.no_automatic_dependency_tracking
    def __init__(self, cls_ref, method_name, **kwargs):
        self.cls_ref = cls_ref
        self.method_name = method_name
        self.cls_symbol = get_canonical_name_for_symbol(self.cls_ref, add_prefix_to_v1_names=True) or get_canonical_name_for_symbol(self.cls_ref, api_name='keras', add_prefix_to_v1_names=True)
        if 'name' not in kwargs:
            kwargs['name'] = K.unique_object_name('tf.' + self.cls_symbol + '.' + self.method_name, zero_based=True, avoid_observed_names=True)
        kwargs['autocast'] = False
        self._must_restore_from_config = True
        super(ClassMethod, self).__init__(**kwargs)
        self._preserve_input_structure_in_config = True
        self._expects_training_arg = False
        self._expects_mask_arg = False

    def call(self, args, kwargs):
        return getattr(self.cls_ref, self.method_name)(*args, **kwargs)

    def get_config(self):
        if not self.cls_symbol:
            raise ValueError('This Keras class method conversion tried to convert a method belonging to class %s, a class that is not an exposed in the TensorFlow API. To ensure cross-version compatibility of Keras models that use op layers, only op layers produced from exported TF API symbols can be serialized.' % self.cls_symbol)
        config = {'cls_symbol': self.cls_symbol, 'method_name': self.method_name}
        base_config = super(ClassMethod, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        symbol_name = config.pop('cls_symbol')
        cls_ref = get_symbol_from_name(symbol_name)
        if not cls_ref:
            raise ValueError('TF symbol `tf.%s` could not be found.' % symbol_name)
        config['cls_ref'] = cls_ref
        return cls(**config)