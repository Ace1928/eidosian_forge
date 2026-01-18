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
def _recreate_function(self, proto, dependencies):
    fn = function_deserialization.recreate_function(proto, self._concrete_functions)
    for name in proto.concrete_functions:
        self._setup_function_captures(name, dependencies)
    if self._save_options.experimental_load_function_aliases:
        if proto.concrete_functions and all((name in self._concrete_function_aliases for name in proto.concrete_functions)):
            alias = self._concrete_function_aliases[next(iter(proto.concrete_functions))]
            aliased = self.function_aliases.get(alias)
            assert isinstance(aliased, list)
            if set((f.name for f in aliased)) == set((f.name for f in fn._list_all_concrete_functions())):
                self.function_aliases[alias] = fn
            else:
                logging.warn("Not aliasing '%s' to polymorphic restored function because of mismatched concrete functions: %s vs %s", alias, set((f.name for f in aliased)), set((f.name for f in fn._list_all_concrete_functions())))
    return (fn, setattr)