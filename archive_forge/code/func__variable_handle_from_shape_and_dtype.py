import contextlib
import functools
import weakref
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager import tape
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_module
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_resource_variable_ops import *
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _variable_handle_from_shape_and_dtype(shape, dtype, shared_name, name, graph_mode, initial_value=None):
    """Create a variable handle, copying in handle data from `initial_value`."""
    container = ops.get_default_graph()._container
    if container is None:
        container = ''
    shape = tensor_shape.as_shape(shape)
    dtype = dtypes.as_dtype(dtype)
    if not graph_mode:
        if shared_name is not None:
            raise errors.InternalError(node_def=None, op=None, message='Using an explicit shared_name is not allowed when executing eagerly.')
        shared_name = context.anonymous_name()
    handle = gen_resource_variable_ops.var_handle_op(shape=shape, dtype=dtype, shared_name=shared_name, name=name, container=container)
    if initial_value is None:
        initial_value = handle
    if graph_mode:
        full_handle_data = _combine_handle_data(handle, initial_value)
        _set_handle_shapes_and_types(handle, full_handle_data, graph_mode)
        return handle
    else:
        handle_data = handle_data_util.create_handle_data(shape, dtype)
        if initial_value is not None and initial_value.dtype == dtypes.variant:
            extra_handle_data = get_eager_safe_handle_data(initial_value)
            if extra_handle_data is not None and extra_handle_data.is_set:
                if not handle_data.is_set or len(handle_data.shape_and_type) != 1:
                    raise RuntimeError(f"Expected VarHandleOp to return a length==1 shape_and_type, but saw: '{handle_data}'")
                handle_data.shape_and_type.extend(extra_handle_data.shape_and_type)
        _set_handle_shapes_and_types(handle, handle_data, graph_mode)
        return handle