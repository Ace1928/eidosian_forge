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
def _retrieve_all_filtered_nodes(self):
    """Traverses through the object graph to get the IDs of all nodes to load.

    As a side-effect, if node_filters is a dictionary that contains already-
    created objects, then the children tracked by those objects will be
    added to node_filters.

    Returns:
      List of all nodes to load, or None if all nodes should be loaded.

    """
    if self._node_filters is None:
        return None
    all_filtered_nodes = set()
    nodes_to_visit = list(self._node_filters)
    while nodes_to_visit:
        node_path = nodes_to_visit.pop(0)
        node_id = self._node_path_to_id[node_path]
        if node_id in all_filtered_nodes:
            continue
        all_filtered_nodes.add(node_id)
        node, setter = self._loaded_nodes.get(node_id, (None, None))
        if node is not None:
            if not isinstance(node, base.Trackable):
                raise TypeError(f"Error when processing dictionary values passed to nodes_to_load.Object at {node_path} is expected to be a checkpointable (i.e. 'trackable') TensorFlow object (e.g. tf.Variable, tf.Module or Keras layer).")
            node._maybe_initialize_trackable()
        for reference in self._proto.nodes[node_id].children:
            child_object, _ = self._loaded_nodes.get(reference.node_id, (None, None))
            if child_object is None and node is not None:
                child_object = node._lookup_dependency(reference.local_name)
                if isinstance(child_object, data_structures.TrackableDataStructure):
                    setter = lambda *args: None
                    self._loaded_nodes[reference.node_id] = (child_object, setter)
            child_path = '{}.{}'.format(node_path, reference.local_name)
            self._node_path_to_id[child_path] = reference.node_id
            nodes_to_visit.append(child_path)
    if 0 in all_filtered_nodes:
        return None
    return all_filtered_nodes