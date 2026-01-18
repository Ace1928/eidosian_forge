import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
def _create_grad_func(ys, xs, grads, cond_graph, body_graph, name, while_op, maximum_iterations):
    """Builds and returns the gradient FuncGraph of `func_graph` and its args.

  The returned grad_func_graph must be called with the returned
  args + grad_func_graph.captures.

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grads: The incoming grads for `ys`.
    cond_graph: FuncGraph for the forward cond function.
    body_graph: FuncGraph for the forward body function.
    name: Name of the returned gradient function.
    while_op: The forward While op.
    maximum_iterations: Tensor. The maximum number of iterations.

  Returns:
    2-tuple of (grad_func_graph, args).
  """
    assert len(ys) == len(grads)
    total_iters = while_op.outputs[0]
    counter = constant_op.constant(0, dtype=total_iters.dtype, name='grad_counter')
    body_graph_inputs = object_identity.ObjectIdentitySet(body_graph.inputs)
    body_graph_outputs = object_identity.ObjectIdentitySet(body_graph.outputs)
    args = [counter, maximum_iterations, total_iters] + list(grads)
    grad_func_graph = func_graph_module.func_graph_from_py_func(name, lambda *args: _grad_fn(ys, xs, args, body_graph), args, {}, func_graph=_WhileBodyGradFuncGraph(name, cond_graph, body_graph, maximum_iterations, while_op, body_graph_inputs, body_graph_outputs))
    for external_capture, internal_capture in grad_func_graph.captures:
        if ops.tensor_id(internal_capture) in grad_func_graph.internal_capture_to_output:
            new_output = grad_func_graph.internal_capture_to_output[ops.tensor_id(internal_capture)]
        else:
            raise ValueError(f'Tensor {str(internal_capture)} which captures {str(external_capture)} is in list of internal_captures but not in internal_capture_to_output.')
        grad_func_graph.outputs.append(new_output)
        grad_func_graph.structured_outputs.append(new_output)
    return (grad_func_graph, args)