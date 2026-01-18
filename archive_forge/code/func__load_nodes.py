import collections
import functools
import os
import sys
from absl import logging
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.capture import restore_captures
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import restore
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager.polymorphic_function import saved_model_utils as function_saved_model_utils
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _load_nodes(self):
    """Load all saved objects."""
    nodes, node_setters = self._initialize_loaded_nodes()
    slot_variable_node_ids = {}
    for node_id, proto in self._iter_all_nodes():
        for slot_variable_proto in proto.slot_variables:
            slot_variable_node_id = slot_variable_proto.slot_variable_node_id
            slot_variable_node_ids[slot_variable_node_id] = (node_id, slot_variable_proto)
    for node_id, proto in self._iter_all_nodes():
        if nodes.get(node_id) is not None:
            continue
        elif node_id in slot_variable_node_ids:
            optimizer_node_id, slot_variable_proto = slot_variable_node_ids[node_id]
            optimizer_object = nodes[optimizer_node_id]
            optimized_variable = nodes[slot_variable_proto.original_variable_node_id]
            slot_variable = optimizer_object.add_slot(var=optimized_variable, slot_name=slot_variable_proto.slot_name)
            nodes[slot_variable_proto.slot_variable_node_id] = slot_variable
            node_setters[slot_variable_proto.slot_variable_node_id] = setattr
        else:
            node, setter = self._recreate(proto, node_id, nodes)
            nodes[node_id] = node
            node_setters[node_id] = setter
    if 0 not in nodes:
        nodes[0] = self._recreate_base_user_object()[0]
    self._nodes = [nodes.get(node_id) for node_id in range(len(self._proto.nodes))]
    self._node_setters = node_setters