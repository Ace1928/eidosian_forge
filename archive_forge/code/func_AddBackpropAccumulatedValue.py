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
def AddBackpropAccumulatedValue(self, history_value, value, dead_branch=False):
    """Add the getter for an accumulated value in the grad context.

    This is added to the backprop loop. Called in the grad context to
    get the value of an accumulated value. The stack pop op must be guarded
    by the pred of the controlling cond.

    Args:
      history_value: The history (a stack) of a value.
      value: The value that is pushed onto the stack.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The current value (the top of the stack).
    """
    history_ctxt = history_value.op._get_control_flow_context()
    cond_ctxt = None
    value_ctxt = value.op._get_control_flow_context()
    while value_ctxt and value_ctxt != history_ctxt:
        if isinstance(value_ctxt, control_flow_ops.CondContext):
            cond_ctxt = value_ctxt
            break
        value_ctxt = value_ctxt.outer_context
    with ops.control_dependencies(None):
        self.grad_context.Enter()
        if cond_ctxt:
            grad_state = self
            pred = None
            while pred is None and grad_state:
                pred = grad_state.history_map.get(cond_ctxt.pred.name)
                grad_state = grad_state.outer_grad_state
            if pred is None:
                pred = cond_ctxt.pred
            branch = 1 - cond_ctxt.branch if dead_branch else cond_ctxt.branch
            history_value = control_flow_ops._SwitchRefOrTensor(history_value, pred)[branch]
        pop = gen_data_flow_ops.stack_pop_v2(history_value, value.dtype.base_dtype)
        pop.set_shape(value.get_shape())
        self.grad_context.Exit()
    parallel_iterations = self.grad_context.parallel_iterations
    if parallel_iterations > 1:
        self.grad_sync._add_control_input(pop.op)
    return pop