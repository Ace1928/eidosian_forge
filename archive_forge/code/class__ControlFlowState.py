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
class _ControlFlowState:
    """Maintain the mapping from the loops to their grad states."""

    def __init__(self):
        self._map = {}

    def GetGradState(self, op, before):
        """Return the grad state for this op if it's in a forward loop context."""
        if before and util.IsLoopExit(op):
            forward_ctxt = op._get_control_flow_context()
            forward_ctxt = forward_ctxt.outer_context
            if forward_ctxt:
                forward_ctxt = forward_ctxt.GetWhileContext()
        else:
            forward_ctxt = util.GetWhileContext(op)
        if forward_ctxt:
            return self._map.get(forward_ctxt)
        return None

    def ProcessUnusedLoopExits(self, pending_count, to_ops_set):
        """Process all the "unused" loop exits.

    The "unused" exits of the loops are added to `unused_exits`. An exit is
    unused if its pending_count is 0. If there is an exit with real gradient,
    all these deferred exits will enter the backprop loop with zero gradient.
    Otherwise, they will enter the backprop loop with None. As an example,
    people often write:

    ```python
    v1, _ = tf.while_loop(p, b, [x1, x2])
    result = gradients(v1, x1)
    ```

    The exit node for x2 is not included by the betweenness analysis. But we
    need to backprop x2 if x2 is involved in computing v1.

    Args:
      pending_count: The number of backprop inputs for every op.
      to_ops_set: The set of ops for ys in gradients(ys, xs)

    Returns:
      The set of unused loop exits that we know at this point we need
      to backprop.
    """
        loop_exits = []
        for grad_state in self._map.values():
            for y in grad_state.forward_loop_exits:
                if pending_count[y.op] == 0:
                    grad_state.pending_exits_count -= 1
                    if y.op not in to_ops_set:
                        grad_state.unused_exits.append(y)
                    if grad_state.pending_exits_count == 0:
                        loop_exits.extend(grad_state.unused_exits)
            for y in grad_state.forward_context.loop_enters:
                if pending_count[y.op] == 0:
                    pending_count[y.op] = 1
        return loop_exits

    def EnterGradWhileContext(self, op, before):
        """Enter the WhileContext for gradient computation."""
        grad_state = self.GetGradState(op, before)
        if grad_state:
            grad_state.grad_context.Enter()

    def ExitGradWhileContext(self, op, before):
        """Exit the WhileContext for gradient computation."""
        grad_state = self.GetGradState(op, before)
        if grad_state:
            grad_state.grad_context.Exit()

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

    def ZerosLikeForExit(self, val):
        """Create zeros_like gradient for a loop exit.

    If the result of a loop variable is not used but is involved in
    computing the result of some needed loop variable, we create a
    zero-valued tensor that is fed as gradient for the Exit node of that
    loop variable. Note that val.op is an Exit, and this method must be
    called in the control flow context where gradients() is called.

    Args:
      val: The output tensor of an Exit op.

    Returns:
      A zero tensor of the same shape of val.
    """
        val_shape = val.get_shape()
        forward_ctxt = val.op._get_control_flow_context()
        outer_forward_ctxt = forward_ctxt.outer_context
        if outer_forward_ctxt:
            outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
        outer_grad_state = None
        if outer_forward_ctxt:
            outer_grad_state = self._map.get(outer_forward_ctxt)
        if outer_grad_state:
            if val_shape.is_fully_defined():
                outer_grad_state.grad_context.Enter()
                result = array_ops.zeros(val_shape.dims, val.dtype)
                outer_grad_state.grad_context.Exit()
            else:
                forward_ctxt.outer_context.Enter()
                shape = array_ops.shape_internal(val, optimize=False)
                forward_ctxt.outer_context.Exit()
                history_shape = outer_grad_state.AddForwardAccumulator(shape)
                outer_grad_ctxt = outer_grad_state.grad_context
                outer_grad_ctxt.Enter()
                real_shape = outer_grad_state.AddBackpropAccumulatedValue(history_shape, shape)
                result = array_ops.zeros(real_shape, val.dtype)
                outer_grad_ctxt.Exit()
        elif val_shape.is_fully_defined():
            result = array_ops.zeros(val_shape.dims, val.dtype)
        else:
            result = array_ops.zeros_like(val, optimize=False)
        return result

    def ZerosLikeV1WhileLoop(self, op, index):
        """Create zeros_like for the specified output of an op.

    If op is in a while loop that is part of gradients(), this method
    must be called in its grad loop context.

    Args:
      op: A tensorflow operation.
      index: the index for a specific output of the op.

    Returns:
      A zero tensor of the same shape of op.outputs[index].
    """
        if util.IsLoopSwitch(op):
            return None
        if op.graph.building_function:
            return array_ops.zeros_like(op.outputs[index])
        dead_branch = util.IsSwitch(op)
        forward_ctxt = util.GetWhileContext(op)
        grad_state = self._map.get(forward_ctxt)
        if grad_state is None:
            return ZerosLike(op, index)
        op_ctxt = op._get_control_flow_context()
        val = ops.convert_to_tensor(op.outputs[index], name='tensor')
        shape = val.get_shape()
        if shape.is_fully_defined():
            if val.dtype == dtypes.resource:
                result = array_ops.zeros(resource_variable_ops.variable_shape(val), dtype=default_gradient.get_zeros_dtype(val))
            else:
                result = constant_op.constant(0, shape=shape.dims, dtype=val.dtype)
            if dead_branch:
                pred = grad_state.history_map.get(op_ctxt.pred.name)
                branch = op_ctxt.branch
                result = control_flow_ops._SwitchRefOrTensor(result, pred)[1 - branch]
        else:
            if dead_branch:
                pred = op_ctxt.pred
                branch = op_ctxt.branch
                op_ctxt.outer_context.Enter()
                val = control_flow_ops._SwitchRefOrTensor(op.inputs[0], pred)[1 - branch]
                zeros_shape = array_ops.shape_internal(val, optimize=False)
                op_ctxt.outer_context.Exit()
                val.op._set_control_flow_context(op_ctxt)
                zeros_shape.op._set_control_flow_context(op_ctxt)
            else:
                op_ctxt.Enter()
                zeros_shape = array_ops.shape_internal(val, optimize=False)
                op_ctxt.Exit()
            grad_state.grad_context.Exit()
            history_zeros_shape = grad_state.AddForwardAccumulator(zeros_shape, dead_branch=dead_branch)
            grad_state.grad_context.Enter()
            shape = grad_state.AddBackpropAccumulatedValue(history_zeros_shape, zeros_shape, dead_branch)
            result = array_ops.zeros(shape, val.dtype)
        return result

    def PostProcessing(self):
        """Perform postprocessing at the end of gradients().

    We have created the gradient graph at this point. So this function
    can be used to perform any postprocessing on the gradient graph.
    We currently perform the following postprocessing:
      1. Patch the gradient graph if the output of a loop variable
         doesn't depend on its input.
    """
        for _, grad_state in self._map.items():
            for _, b_merge in grad_state.switch_map.items():
                if b_merge.op.inputs[0] == b_merge.op.inputs[1]:
                    dtype = b_merge.op.inputs[0].dtype
                    shape = b_merge.op.inputs[0].get_shape()
                    if shape.is_fully_defined():
                        grad_state.grad_context.Enter()
                        grad_val = constant_op.constant(0, dtype=dtype, shape=shape)
                        next_grad_val = control_flow_ops._NextIteration(grad_val)
                        grad_state.grad_context.Exit()
                    else:
                        outer_grad_ctxt = grad_state.grad_context.outer_context
                        if outer_grad_ctxt:
                            outer_grad_ctxt.Enter()
                        enter_grad_op = b_merge.op.inputs[0].op
                        enter_grad = enter_grad_op.inputs[0]
                        grad_shape = array_ops.shape_internal(enter_grad, optimize=False)
                        grad_val = array_ops.zeros(grad_shape)
                        if outer_grad_ctxt:
                            outer_grad_ctxt.Exit()
                        grad_state.grad_context.Enter()
                        next_grad_val = control_flow_ops._NextIteration(grad_val)
                        grad_state.grad_context.Exit()
                    b_merge.op._update_input(1, next_grad_val)