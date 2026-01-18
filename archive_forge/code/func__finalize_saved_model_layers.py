import os
import re
import types
from google.protobuf import message
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.protobuf import saved_metadata_pb2
from tensorflow.python.keras.protobuf import versions_pb2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
def _finalize_saved_model_layers(layers):
    """Runs the final steps of loading Keras Layers from SavedModel."""
    for layer in layers:
        layer.built = True
        layer_call = getattr(_get_keras_attr(layer), 'call_and_return_conditional_losses', None)
        if layer_call and layer_call.concrete_functions:
            layer.call = utils.use_wrapped_call(layer, layer_call, return_method=True)
            expects_training_arg = layer._serialized_attributes['metadata']['expects_training_arg']
            if 'training' in layer_call.function_spec.arg_names:
                expects_training_arg = True
            layer._init_call_fn_args(expects_training_arg)
        else:
            layer.call = types.MethodType(_unable_to_call_layer_due_to_serialization_issue, layer)
    for layer in layers:
        if isinstance(layer, RevivedNetwork):
            _set_network_attributes_from_metadata(layer)
            if hasattr(_get_keras_attr(layer), 'call_and_return_conditional_losses'):
                call_fn = _get_keras_attr(layer).call_and_return_conditional_losses
                if not call_fn.concrete_functions:
                    continue
                if call_fn.input_signature is None:
                    inputs = infer_inputs_from_restored_call_function(call_fn)
                else:
                    inputs = call_fn.input_signature[0]
                layer._set_inputs(inputs)
        _restore_layer_unconditional_losses(layer)
        _restore_layer_activation_loss(layer)
        _restore_layer_metrics(layer)