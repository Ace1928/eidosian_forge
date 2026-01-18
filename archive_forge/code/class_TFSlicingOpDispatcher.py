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
class TFSlicingOpDispatcher(dispatch.OpDispatcher):
    """A global dispatcher that allows building a functional model with TF Ops."""

    def __init__(self, op):
        self.op = op

    def handle(self, args, kwargs):
        """Handle the specified operation with the specified arguments."""
        args = nest.map_structure(_slice_to_dict, args)
        kwargs = nest.map_structure(_slice_to_dict, kwargs)
        if any((isinstance(x, keras_tensor.KerasTensor) for x in nest.flatten([args, kwargs]))):
            return SlicingOpLambda(self.op)(*args, **kwargs)
        else:
            return self.NOT_SUPPORTED