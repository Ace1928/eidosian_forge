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
class _UnreadVariable(BaseResourceVariable):
    """Represents a future for a read of a variable.

  Pretends to be the tensor if anyone looks.
  """

    def __init__(self, handle, dtype, shape, in_graph_mode, parent_op, unique_id):
        if isinstance(handle, ops.EagerTensor):
            handle_name = ''
        else:
            handle_name = handle.name
        if context.executing_eagerly() or ops.inside_function():
            graph_element = None
        else:
            with ops.control_dependencies([parent_op]):
                graph_element = gen_resource_variable_ops.read_variable_op(handle, dtype)
                _maybe_set_handle_data(dtype, handle, graph_element)
        super(_UnreadVariable, self).__init__(handle=handle, shape=shape, handle_name=handle_name, unique_id=unique_id, dtype=dtype, graph_element=graph_element)
        self._parent_op = parent_op

    @property
    def name(self):
        if self._in_graph_mode:
            return self._parent_op.name
        else:
            return 'UnreadVariable'

    def value(self):
        return self._read_variable_op()

    def read_value(self):
        return self._read_variable_op()

    def _read_variable_op(self):
        with ops.control_dependencies([self._parent_op]):
            result = gen_resource_variable_ops.read_variable_op(self._handle, self._dtype)
            _maybe_set_handle_data(self._dtype, self._handle, result)
            return result

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).assign_sub(delta, use_locking, name, read_value)

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).assign_add(delta, use_locking, name, read_value)

    def assign(self, value, use_locking=None, name=None, read_value=True):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).assign(value, use_locking, name, read_value)

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_sub(sparse_delta, use_locking, name)

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_add(sparse_delta, use_locking, name)

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_max(sparse_delta, use_locking, name)

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_min(sparse_delta, use_locking, name)

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_mul(sparse_delta, use_locking, name)

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_div(sparse_delta, use_locking, name)

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_update(sparse_delta, use_locking, name)

    def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).batch_scatter_update(sparse_delta, use_locking, name)

    def scatter_nd_sub(self, indices, updates, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_sub(indices, updates, name)

    def scatter_nd_add(self, indices, updates, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_add(indices, updates, name)

    def scatter_nd_update(self, indices, updates, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_update(indices, updates, name)

    def scatter_nd_max(self, indices, updates, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_max(indices, updates, name)

    def scatter_nd_min(self, indices, updates, name=None):
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_min(indices, updates, name)

    @property
    def op(self):
        """The op for this variable."""
        return self._parent_op