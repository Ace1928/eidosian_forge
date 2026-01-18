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
def _recreate_variable(self, proto):
    name = proto.name if proto.name else None
    if name is not None:
        dbg_name = name
    else:
        dbg_name = '<variable loaded from saved model>'
    synchronization, aggregation, trainable = variables.validate_synchronization_aggregation_trainable(proto.synchronization, proto.aggregation, proto.trainable, name=dbg_name)

    def uninitialized_variable_creator(next_creator, **kwargs):
        """A variable creator that creates uninitialized variables."""
        del next_creator
        return resource_variable_ops.UninitializedVariable(**kwargs)
    with ops.get_default_graph()._variable_creator_scope(uninitialized_variable_creator, priority=50):
        saved_device = proto.device
        load_with_device = self._save_options.experimental_variable_policy._save_variable_devices() and config.get_soft_device_placement() and saved_device
        if load_with_device:
            with ops.device(saved_device):
                return (variables.Variable(shape=proto.shape, dtype=proto.dtype, name=name, trainable=trainable, synchronization=synchronization, aggregation=aggregation), setattr)
        else:
            return (variables.Variable(shape=proto.shape, dtype=proto.dtype, name=name, trainable=trainable, synchronization=synchronization, aggregation=aggregation), setattr)