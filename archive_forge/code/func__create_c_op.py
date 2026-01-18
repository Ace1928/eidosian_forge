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
@tf_export('__internal__.create_c_op', v1=[])
@traceback_utils.filter_traceback
def _create_c_op(graph, node_def, inputs, control_inputs, op_def=None, extract_traceback=True):
    """Creates a TF_Operation.

  Args:
    graph: a `Graph`.
    node_def: `node_def_pb2.NodeDef` for the operation to create.
    inputs: A flattened list of `Tensor`s. This function handles grouping
      tensors into lists as per attributes in the `node_def`.
    control_inputs: A list of `Operation`s to set as control dependencies.
    op_def: Optional. `op_def_pb2.OpDef` for the operation to create. If not
      specified, is looked up from the `graph` using `node_def.op`.
    extract_traceback: if True, extract the current Python traceback to the
      TF_Operation.

  Returns:
    A wrapped TF_Operation*.
  """
    if op_def is None:
        op_def = graph.op_def_for_type(node_def.op)
    inputs = _reconstruct_sequence_inputs(op_def, inputs, node_def.attr)
    with graph._c_graph.get() as c_graph:
        op_desc = pywrap_tf_session.TF_NewOperation(c_graph, compat.as_str(node_def.op), compat.as_str(node_def.name))
    if node_def.device:
        pywrap_tf_session.TF_SetDevice(op_desc, compat.as_str(node_def.device))
    for op_input in inputs:
        if isinstance(op_input, (list, tuple)):
            pywrap_tf_session.TF_AddInputList(op_desc, [t._as_tf_output() for t in op_input])
        else:
            pywrap_tf_session.TF_AddInput(op_desc, op_input._as_tf_output())
    for control_input in control_inputs:
        pywrap_tf_session.TF_AddControlInput(op_desc, control_input._c_op)
    for name, attr_value in node_def.attr.items():
        serialized = attr_value.SerializeToString()
        pywrap_tf_session.TF_SetAttrValueProto(op_desc, compat.as_str(name), serialized)
    try:
        c_op = pywrap_tf_session.TF_FinishOperation(op_desc)
    except errors.InvalidArgumentError as e:
        raise ValueError(e.message)
    if extract_traceback:
        pywrap_tf_session.TF_SetOpStackTrace(c_op, tf_stack.extract_stack(stacklevel=3))
    return c_op