import collections
import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple
from absl import logging
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_fn
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import pywrap_saved_model
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import tracing_utils
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import trace_saveable_util
from tensorflow.python.types import core as types_core
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def _trace_gradient_functions(graph: ops.Graph, saveable_view: _SaveableView):
    """Traces gradient functions and records them in the SaveableView."""
    functions = list(graph._functions.values())
    func_graph_map = {f.graph: f for f in functions if hasattr(f, 'graph')}
    seen_op_types = set()
    for fn in functions:
        for op_type, op in _iterate_op_types(fn):
            if op_type in seen_op_types:
                continue
            seen_op_types.add(op_type)
            try:
                custom_gradient = ops.gradient_registry.lookup(op_type)
            except LookupError:
                continue
            try:
                grad_fn = def_function.function(custom_gradient).get_concrete_function(None, *op.inputs)
            except Exception as exc:
                traceback.print_exc()
                raise ValueError(f'Error when tracing gradients for SavedModel.\n\nCheck the error log to see the error that was raised when converting a gradient function to a concrete function. You may need to update the custom gradient, or disable saving gradients with the option tf.saved_model.SaveOptions(experimental_custom_gradients=False).\n\tProblematic op name: {op.name}\n\tGradient inputs: {op.inputs}') from exc
            with graph.as_default():
                bad_captures = []
                for capture in grad_fn.captured_inputs:
                    if capture.dtype in _UNCOPIABLE_DTYPES:
                        continue
                    outer_fn, outer_capture = _get_outer_most_capture(fn, capture, func_graph_map)
                    if outer_fn is None or isinstance(outer_capture, ops.EagerTensor):
                        if outer_capture not in saveable_view.captured_tensor_node_ids:
                            raise ValueError(f'Found invalid capture {outer_capture} when saving custom gradients.')
                        saveable_view.captured_tensor_node_ids[capture] = saveable_view.captured_tensor_node_ids[outer_capture]
                    elif outer_capture.graph is outer_fn.graph:
                        capture_name = outer_capture.name
                        if isinstance(outer_fn, defun.AtomicFunction):
                            try:
                                arg_index = outer_fn.graph.inputs.index(outer_capture)
                                capture_name = outer_fn.cached_definition.signature.input_arg[arg_index].name + ':0'
                            except ValueError:
                                pass
                        node = _CapturedTensor(capture_name, outer_fn.name)
                        saveable_view.add_capture_and_node(capture, node)
                    else:
                        bad_captures.append(capture.name)
                if not bad_captures:
                    grad_fn.add_to_graph(graph)
                else:
                    raise ValueError(f'Cannot save custom gradient {op_type} called in function {fn} because SavedModel is unable to serialize the captured inputs: {bad_captures}')
            saveable_view.gradient_functions.append(grad_fn)
            func_graph_map[grad_fn.graph] = grad_fn
            grad_def = function_pb2.RegisteredGradient()
            grad_def.gradient_func = grad_fn.name
            grad_def.registered_op_type = op_type
            saveable_view.gradient_defs.append(grad_def)