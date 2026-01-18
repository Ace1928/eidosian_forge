import functools
import operator
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import imperative_grad
from tensorflow.python.eager import tape
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _gradient_function(op_name, attr_tuple, num_inputs, inputs, outputs, out_grads, skip_input_indices, forward_pass_name_scope):
    """Calls the gradient function of the op.

  Args:
    op_name: the name of the op to be differentiated.
    attr_tuple: the attrs, as a tuple.
    num_inputs: the number of inputs to the op.
    inputs: inputs to the original operation.
    outputs: outputs to the original operation.
    out_grads: gradients of the operation wrt its outputs.
    skip_input_indices: a tuple that is passed to the gradient function,
      indicating which inputs to skip calculating the gradient for
    forward_pass_name_scope: the namescope of the op in the forward pass.

  Returns:
    The gradients with respect to the inputs of the function, as a list.
  """
    mock_op = _MockOp(attr_tuple, inputs, outputs, op_name, skip_input_indices)
    grad_fn = ops._gradient_registry.lookup(op_name)
    if grad_fn is None:
        return [None] * num_inputs
    if ops.executing_eagerly_outside_functions() or control_flow_util.EnableControlFlowV2(ops.get_default_graph()):
        gradient_name_scope = 'gradient_tape/'
        if forward_pass_name_scope:
            gradient_name_scope += forward_pass_name_scope + '/'
        with ops.name_scope(gradient_name_scope):
            return grad_fn(mock_op, *out_grads)
    else:
        return grad_fn(mock_op, *out_grads)