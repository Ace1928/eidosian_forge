import collections
import copy
import itertools
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
@trackable.no_automatic_dependency_tracking
def _init_graph_network(self, inputs, outputs):
    self._is_graph_network = True
    if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
        inputs = inputs[0]
    if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
        outputs = outputs[0]
    self._nested_inputs = inputs
    self._nested_outputs = outputs
    self.inputs = nest.flatten(inputs)
    self.outputs = nest.flatten(outputs)
    if not nest.is_nested(self._nested_inputs):
        self._enable_dict_to_input_mapping = True
    elif isinstance(self._nested_inputs, (list, tuple)) and (not any((nest.is_nested(t) for t in self._nested_inputs))):
        self._enable_dict_to_input_mapping = True
    elif isinstance(self._nested_inputs, dict) and (not any((nest.is_nested(t) for t in self._nested_inputs.values()))):
        self._enable_dict_to_input_mapping = True
    else:
        self._enable_dict_to_input_mapping = False
    if not ops.executing_eagerly_outside_functions():
        if any((not hasattr(tensor, '_keras_history') for tensor in self.outputs)):
            base_layer_utils.create_keras_history(self._nested_outputs)
    self._validate_graph_inputs_and_outputs()
    self.built = True
    self._build_input_shape = nest.map_structure(lambda x: x.shape, inputs)
    self._compute_output_and_mask_jointly = True
    self._expects_training_arg = True
    self._expects_mask_arg = True
    self._autocast = False
    self._input_layers = []
    self._output_layers = []
    self._input_coordinates = []
    self._output_coordinates = []
    self._output_mask_cache = {}
    self._output_tensor_cache = {}
    self._output_shape_cache = {}
    for x in self.outputs:
        layer, node_index, tensor_index = x._keras_history
        self._output_layers.append(layer)
        self._output_coordinates.append((layer, node_index, tensor_index))
    for x in self.inputs:
        layer, node_index, tensor_index = x._keras_history
        assert node_index == 0
        assert tensor_index == 0
        self._input_layers.append(layer)
        self._input_coordinates.append((layer, node_index, tensor_index))
    nodes, nodes_by_depth, layers, _ = _map_graph_network(self.inputs, self.outputs)
    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._self_tracked_trackables = layers
    self._layer_call_argspecs = {}
    for layer in self._self_tracked_trackables:
        self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
    self._set_output_names()
    self.input_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for layer in self._input_layers:
        self.input_names.append(layer.name)
        if layer.is_placeholder:
            self._feed_input_names.append(layer.name)
            self._feed_input_shapes.append(layer._batch_input_shape)
            self._feed_inputs.append(layer.input)
    self._compute_tensor_usage_count()
    self._set_save_spec(self._nested_inputs)
    tf_utils.assert_no_legacy_layers(self.layers)