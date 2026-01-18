import collections
import copy
import enum
import re
import sys
import threading
import types
from typing import Any, AnyStr, Callable, List, NoReturn, Pattern, Tuple, Type, Union, Optional
from absl import app
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import registry
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import versions
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace as profiler_trace
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lock_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_stack
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import kwarg_only
from tensorflow.python.util.tf_export import tf_export
class _ControlDependenciesController(object):
    """Context manager for `control_dependencies()`."""

    def __init__(self, graph, control_inputs):
        """Create a new `_ControlDependenciesController`.

      A `_ControlDependenciesController` is the context manager for
      `with tf.control_dependencies()` blocks.  These normally nest,
      as described in the documentation for `control_dependencies()`.

      The `control_inputs` argument list control dependencies that must be
      added to the current set of control dependencies.  Because of
      uniquification the set can be empty even if the caller passed a list of
      ops.  The special value `None` indicates that we want to start a new
      empty set of control dependencies instead of extending the current set.

      In that case we also clear the current control flow context, which is an
      additional mechanism to add control dependencies.

      Args:
        graph: The graph that this controller is managing.
        control_inputs: List of ops to use as control inputs in addition to the
          current control dependencies.  None to indicate that the dependencies
          should be cleared.
      """
        self._graph = graph
        if control_inputs is None:
            self._control_inputs_val = []
            self._new_stack = True
        else:
            self._control_inputs_val = control_inputs
            self._new_stack = False
        self._seen_nodes = set()
        self._old_stack = None
        self._old_control_flow_context = None

    def __enter__(self):
        if self._new_stack:
            self._old_stack = self._graph._control_dependencies_stack
            self._graph._control_dependencies_stack = []
            self._old_control_flow_context = self._graph._get_control_flow_context()
            self._graph._set_control_flow_context(None)
        self._graph._push_control_dependencies_controller(self)

    def __exit__(self, unused_type, unused_value, unused_traceback):
        self._graph._pop_control_dependencies_controller(self)
        if self._new_stack:
            self._graph._control_dependencies_stack = self._old_stack
            self._graph._set_control_flow_context(self._old_control_flow_context)

    @property
    def control_inputs(self):
        return self._control_inputs_val

    def add_op(self, op):
        if isinstance(op, tensor_lib.Tensor):
            op = op.ref()
        self._seen_nodes.add(op)

    def op_in_group(self, op):
        if isinstance(op, tensor_lib.Tensor):
            op = op.ref()
        return op in self._seen_nodes