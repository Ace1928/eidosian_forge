from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
def AddWhileContext(self, op, between_op_list, between_ops):
    """Add the grad state for the while loop that op belongs to.

    Note that op is an Exit, and this method must be called in
    the control flow context where gradients() is called.

    Note that this method modifies `between_op_list` and `between_ops`.
    """
    forward_ctxt = util.GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
        outer_forward_ctxt = forward_ctxt.outer_context
        if outer_forward_ctxt:
            outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
        outer_grad_state = None
        if outer_forward_ctxt:
            outer_grad_state = self._map.get(outer_forward_ctxt)
        grad_state = _GradLoopState(forward_ctxt, outer_grad_state)
        self._map[forward_ctxt] = grad_state
        for loop_exit in grad_state.forward_loop_exits:
            if loop_exit.op not in between_ops:
                between_ops.add(loop_exit.op)
                between_op_list.append(loop_exit.op)