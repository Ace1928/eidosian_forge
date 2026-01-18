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
def _generate_ordered_node_ids(self):
    """Orders the node ids so that dependencies appear first."""
    if self._filtered_nodes is None:
        unordered_ids = range(len(self._proto.nodes))
    else:
        unordered_ids = list(self._filtered_nodes)
    dependency_map = collections.defaultdict(list)
    for node_id in unordered_ids:
        deps = dependency_map[node_id]
        if self._loaded_nodes.get(node_id) is not None:
            continue
        proto = self._proto.nodes[node_id]
        for dep in set(self._get_node_dependencies(proto).values()):
            deps.append(dep)
            if self._filtered_nodes is not None and dep not in self._filtered_nodes:
                raise ValueError(f'Unable to partially load SavedModel since the specified filter does not include all required objects for loading (e.g. variables used in functions or deserialization dependencies). Please include this path in the filter: {self._pretty_printer.node_names[dep]}')
        prev_slot = None
        for slot_variable_proto in proto.slot_variables:
            slot_variable_node_id = slot_variable_proto.slot_variable_node_id
            slot_deps = dependency_map[slot_variable_node_id]
            slot_deps.append(node_id)
            slot_deps.append(slot_variable_proto.original_variable_node_id)
            if prev_slot is not None:
                slot_deps.append(prev_slot)
            prev_slot = slot_variable_node_id
    try:
        return list(trackable_utils.order_by_dependency(dependency_map))
    except trackable_utils.CyclicDependencyError:
        raise ValueError('Encountered a cycle in the deserialization dependenciesin the SavedModel. This is extremely unexpected, pleasefile a bug and make sure you are not manually modifying the SavedModel.')