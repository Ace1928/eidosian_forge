import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
class _FuncGraph(ops.Graph):
    """A helper for constructing a function.

  _FuncGraph overrides ops.Graph's create_op() so that we can keep
  track of all inputs into every op created inside the function.  If
  any input is from other graphs, we keep track of it in self.capture
  and substitute the input with a place holder.

  Each captured input's corresponding place holder is converted into a
  function argument and the caller passes in the captured tensor.
  """

    def __init__(self, name, capture_by_value, allowlisted_stateful_ops, capture_resource_var_by_value, *args, **kwargs):
        super(_FuncGraph, self).__init__(*args, **kwargs)
        self._capture_by_value = capture_by_value
        self._allowlisted_stateful_ops = allowlisted_stateful_ops
        self._capture_resource_var_by_value = capture_resource_var_by_value
        self._building_function = True
        self._outer_graph = ops.get_default_graph()
        self._vscope = vs.get_variable_scope()
        self._old_custom_getter = self._vscope.custom_getter
        self.name = name
        self.inputs = []
        self.outputs = []
        self._captured = {}
        self.extra_inputs = []
        self.extra_args = []
        self.extra_vars = []

    @property
    def outer_graph(self):
        """The graph active when this _FuncGraph was created."""
        return self._outer_graph

    @tf_contextlib.contextmanager
    def container(self, container_name):
        """Returns a context manager that specifies the resource container to use.

    Overridden from `tf.Graph` to update both the init_scope container
    and the present inner container. This is necessary to make sure setting
    containers applies correctly both to created variables and to stateful
    ops.

    Args:
      container_name: container name string.

    Returns:
      A context manager for defining resource containers for stateful ops,
        yields the container name.
    """
        original_container = self._container
        with ops.init_scope():
            original_init_container = ops.get_default_graph()._container
        try:
            self._container = container_name
            with ops.init_scope():
                ops.get_default_graph()._container = container_name
            yield self._container
        finally:
            self._container = original_container
            with ops.init_scope():
                ops.get_default_graph()._container = original_init_container

    def getvar(self, getter, name, shape=None, dtype=None, initializer=None, reuse=None, trainable=True, collections=None, use_resource=None, **kwargs):
        """A custom variable getter."""
        with self._outer_graph.as_default():
            var = self._vscope.get_variable(vs._get_default_variable_store(), name, shape=shape, dtype=dtype, initializer=initializer, reuse=reuse, trainable=trainable, collections=collections, use_resource=use_resource)
            self.extra_vars.append(var)
            if isinstance(var, resource_variable_ops.BaseResourceVariable) and self._capture_resource_var_by_value:
                return var.value()
            return var

    def _create_op_internal(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        for i, x in enumerate(inputs):
            if isinstance(x, ops.EagerTensor) or x.graph is not self:
                inputs[i] = self.capture(x)
        return super(_FuncGraph, self)._create_op_internal(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)

    def capture(self, tensor, name=None):
        """Adds the given tensor to this graph and returns the captured tensor."""
        if tensor.ref() in self._captured:
            return self._captured[tensor.ref()]
        elif self._capture_by_value:
            return self._add_tensor_and_parents(tensor)
        else:
            return self._capture_tensor_as_extra_input(tensor, name)

    @property
    def captures(self):
        """Pairs of tensors and captured tensor."""
        return [(k.deref(), v) for k, v in self._captured.items()]

    def _capture_tensor_as_extra_input(self, tensor, name=None):
        self.extra_inputs.append(tensor)
        with ops.control_dependencies(None):
            ph = array_ops.placeholder(tensor.dtype, shape=tensor.get_shape(), name=name)
        if isinstance(tensor, ops.EagerTensor):
            handle_data = tensor._handle_data
            if handle_data:
                handle_data = handle_data.SerializeToString()
        else:
            with tensor.graph._c_graph.get() as c_graph:
                handle_data = c_api.GetHandleShapeAndType(c_graph, tensor._as_tf_output())
        if handle_data:
            with ph.graph._c_graph.get() as c_graph:
                c_api.SetHandleShapeAndType(c_graph, ph._as_tf_output(), compat.as_bytes(handle_data))
        self.inputs.append(ph)
        self._captured[tensor.ref()] = ph
        self.extra_args.append(ph)
        if _is_guaranteed_const(tensor):
            with ops.control_dependencies(None):
                return array_ops.guarantee_const(ph)
        else:
            return ph

    def _add_tensor_and_parents(self, tensor):
        op = self._add_op_and_parents(tensor.op)
        return op.outputs[tensor.value_index]

    def _add_op_and_parents(self, op):
        op_def = graph_to_function_def._get_op_def(op)
        if op._is_stateful and op not in self._allowlisted_stateful_ops:
            raise ValueError(f'Cannot capture a stateful node (name:{op.name}, type:{op.type}) by value.')
        elif op.type in ('Placeholder', 'PlaceholderV2'):
            raise ValueError(f'Cannot capture a placeholder (name:{op.name}, type:{op.type}) by value.')
        captured_inputs = [self._add_tensor_and_parents(x) for x in op.inputs]
        captured_op = self._create_op_internal(op.type, captured_inputs, [o.dtype for o in op.outputs], name=op.name, attrs=op.node_def.attr, op_def=op_def)
        for t, captured_t in zip(op.outputs, captured_op.outputs):
            self._captured[t.ref()] = captured_t
        return captured_op