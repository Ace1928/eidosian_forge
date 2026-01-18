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
class TFOpLambda(Layer):
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
        self.function = function
        self.symbol = get_canonical_name_for_symbol(self.function, add_prefix_to_v1_names=True) or get_canonical_name_for_symbol(self.function, api_name='keras', add_prefix_to_v1_names=True)
        if 'name' not in kwargs:
            if self.symbol:
                name = 'tf.' + self.symbol
            else:
                name = self.function.__name__
            kwargs['name'] = K.unique_object_name(name, zero_based=True, avoid_observed_names=True)
        kwargs['autocast'] = False

        def _call_wrapper(*args, **kwargs):
            return self._call_wrapper(*args, **kwargs)
        self.call = tf_decorator.make_decorator(function, _call_wrapper)
        self._must_restore_from_config = True
        super(TFOpLambda, self).__init__(**kwargs)
        self._preserve_input_structure_in_config = True
        self._already_warned = False
        self._expects_training_arg = False
        self._expects_mask_arg = False

    def _call_wrapper(self, *args, **kwargs):
        created_variables = []

        def _variable_creator(next_creator, **creator_kwargs):
            var = next_creator(**creator_kwargs)
            created_variables.append(var)
            return var
        with backprop.GradientTape(watch_accessed_variables=True) as tape, variable_scope.variable_creator_scope(_variable_creator):
            kwargs.pop('name', None)
            result = self.function(*args, **kwargs)
        self._check_variables(created_variables, tape.watched_variables())
        return result

    def _check_variables(self, created_variables, accessed_variables):
        if not created_variables and (not accessed_variables):
            return
        tracked_weights = set((v.ref() for v in self.weights))
        untracked_new_vars = [v for v in created_variables if v.ref() not in tracked_weights]
        if untracked_new_vars:
            variable_str = '\n'.join(('  {}'.format(i) for i in untracked_new_vars))
            error_str = textwrap.dedent('\n          The following Variables were created within a Lambda layer ({name})\n          but are not tracked by said layer:\n          {variable_str}\n          The layer cannot safely ensure proper Variable reuse across multiple\n          calls, and consquently this behavior is disallowed for safety. Lambda\n          layers are not well suited to stateful computation; instead, writing a\n          subclassed Layer is the recommend way to define layers with\n          Variables.').format(name=self.name, variable_str=variable_str)
            raise ValueError(error_str)
        untracked_used_vars = [v for v in accessed_variables if v.ref() not in tracked_weights]
        if untracked_used_vars and (not self._already_warned):
            variable_str = '\n'.join(('  {}'.format(i) for i in untracked_used_vars))
            self._warn(textwrap.dedent("\n          The following Variables were used a Lambda layer's call ({name}), but\n          are not present in its tracked objects:\n          {variable_str}\n          It is possible that this is intended behavior, but it is more likely\n          an omission. This is a strong indication that this layer should be\n          formulated as a subclassed Layer rather than a Lambda layer.").format(name=self.name, variable_str=variable_str))
            self._already_warned = True

    def _warn(self, msg):
        return tf_logging.warning(msg)

    def get_config(self):
        if not self.symbol:
            raise ValueError('This Keras op layer was generated from %s, a method that is not an exposed in the TensorFlow API. This may have happened if the method was explicitly decorated to add dispatching support, and it was used during Functional model construction. To ensure cross-version compatibility of Keras models that use op layers, only op layers produced from exported TF API symbols can be serialized.' % self.function)
        config = {'function': self.symbol}
        base_config = super(TFOpLambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        symbol_name = config['function']
        function = get_symbol_from_name(symbol_name)
        if not function:
            raise ValueError('TF symbol `tf.%s` could not be found.' % symbol_name)
        config['function'] = function
        return cls(**config)