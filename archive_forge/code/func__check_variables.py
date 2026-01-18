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