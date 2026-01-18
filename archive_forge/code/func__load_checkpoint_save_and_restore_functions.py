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
def _load_checkpoint_save_and_restore_functions(self):
    """Restores the checkpoint-related save/restore functions to all nodes."""
    temp_session = [None]
    for node_id, proto in self._iter_all_nodes():
        node = self.get(node_id)
        if proto.saveable_objects.keys() == {trackable_utils.SERIALIZE_TO_TENSORS_NAME}:
            assert len(proto.saveable_objects) == 1
            saveable_object_proto = next(iter(proto.saveable_objects.values()))
            save_fn_id = saveable_object_proto.save_function
            restore_fn_id = saveable_object_proto.restore_function
            node._serialize_to_tensors = self.get(save_fn_id)
            node._restore_from_tensors = self.get(restore_fn_id)
        else:
            saveable_fn_by_name = {}
            for name, saveable_object_proto in proto.saveable_objects.items():
                save_fn_id = saveable_object_proto.save_function
                restore_fn_id = saveable_object_proto.restore_function
                saveable_fn_by_name[name] = (self.get(save_fn_id), self.get(restore_fn_id))
            node._self_saveable_object_factories = saveable_object_util.recreate_saveable_objects(saveable_fn_by_name, temp_session)