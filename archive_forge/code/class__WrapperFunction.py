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
class _WrapperFunction(function.ConcreteFunction):
    """A class wraps a concrete function to handle different distributed contexts.

  The reason for wrapping a concrete function is because the _captured_inputs
  fields used for in-replica context and cross-replica context are different.
  When `load()` is called from within a tf.distribute.strategy scope, the
  captured inputs are distributed variables. When using these distributed
  variables during calling the function, we need different approaches when it is
  in-replica and when it is not in-replica. When it is in replica, naturally we
  should use the corresponding component of the distributed variable; when it is
  not in-replica, calling the function should mean that it is constructing a
  graph that is not actually going to be used. A typical use case is when
  constructing a functional model. In this case, return a placeholder with a
  control dependency to ensure that is never accessed.
  """

    def __init__(self, concrete_function):
        self.__dict__.update(vars(concrete_function))

    def _call_flat(self, args, captured_inputs):

        def get_handle(x):
            return x.handle if distribute_utils.is_distributed_variable(x) else x

        def get_unused_handle(x):
            return _unused_handle() if distribute_utils.is_distributed_variable(x) else x
        if distribute_lib.get_replica_context() is not None or values_util.is_saving_non_distributed():
            captured_inputs = list(map(get_handle, captured_inputs))
        else:
            captured_inputs = list(map(get_unused_handle, captured_inputs))
        return super()._call_flat(args, captured_inputs)