import collections
import copy
import functools
import itertools
import threading
import warnings
import weakref
import numpy as np
from google.protobuf import json_format
from tensorflow.core.framework import node_def_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import to_snake_case  # pylint: disable=unused-import
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_tensor_list  # pylint: disable=unused-import
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.tools.docs import doc_controls
def _functional_construction_call(self, inputs, args, kwargs, input_list):
    call_context = base_layer_utils.call_context()
    if any((isinstance(x, (np_arrays.ndarray, np.ndarray, float, int)) for x in input_list)):

        def _convert_non_tensor(x):
            if isinstance(x, (np_arrays.ndarray, np.ndarray, float, int)):
                return tensor_conversion.convert_to_tensor_v2_with_dispatch(x)
            return x
        inputs = nest.map_structure(_convert_non_tensor, inputs)
        input_list = nest.flatten(inputs)
    mask_arg_passed_by_framework = False
    input_masks, mask_is_implicit = self._get_input_masks(inputs, input_list, args, kwargs)
    if self._expects_mask_arg and mask_is_implicit:
        kwargs['mask'] = input_masks
        mask_arg_passed_by_framework = True
    training_value = None
    training_arg_passed_by_framework = False
    if self._call_arg_was_passed('training', args, kwargs):
        training_value = self._get_call_arg_value('training', args, kwargs)
        if not self._expects_training_arg:
            kwargs.pop('training')
    if training_value is None:
        if call_context.training is not None:
            training_value = call_context.training
        elif backend.global_learning_phase_is_set():
            training_value = backend.learning_phase()
            if tensor_util.is_tf_type(training_value):
                training_value = math_ops.cast(training_value, dtypes.bool)
            else:
                training_value = bool(training_value)
        else:
            training_value = self._default_training_arg
        if self._expects_training_arg:
            args, kwargs = self._set_call_arg_value('training', training_value, args, kwargs)
            training_arg_passed_by_framework = True
    with call_context.enter(layer=self, inputs=inputs, build_graph=True, training=training_value):
        outputs = self._keras_tensor_symbolic_call(inputs, input_masks, args, kwargs)
        if outputs is None:
            raise ValueError("A layer's `call` method should return a Tensor or a list of Tensors, not None (layer: " + self.name + ').')
        if training_arg_passed_by_framework:
            args, kwargs = self._set_call_arg_value('training', None, args, kwargs, pop_kwarg_if_none=True)
        if mask_arg_passed_by_framework:
            kwargs.pop('mask')
        outputs = self._set_connectivity_metadata((inputs,) + args, kwargs, outputs)
        return outputs