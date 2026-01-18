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
def AddForwardAccumulator(self, value, dead_branch=False):
    """Add an accumulator for each forward tensor that is needed in backprop.

    This is added to the forward loop at the first time when a tensor
    in the forward loop is used by backprop gradient computation loop.
    We create an accumulator that accumulates the value of tensor at each
    iteration. Called in the control flow context where gradients() is called.

    The pseudocode is:
    ```
      acc = stack();
      while (_pivot) {
        acc = stack_push(acc, value);
      }
    ```

    We make sure that the stack push op in one iteration is executed before
    next iteration. This is achieved by adding a control edge from
    `forward_index.op.inputs[0].op` to the push op, and another control
    edge from the push op to either `forward_index.op` or `forward_sync`.

    Args:
      value: The source tensor in forward that is to be accumulated.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The stack that contains the accumulated history of the tensor.

    Raises:
      TypeError: For internal errors involving the value condition context.
      ValueError: If `value` is inside a XLA scope and a valid max size
        for the stack can't be found.
    """
    with self._forward_index.graph.as_default():
        curr_ctxt = ops.get_default_graph()._get_control_flow_context()
        with ops.control_dependencies(None):
            if curr_ctxt:
                curr_ctxt.Enter()
            with ops.colocate_with(value):
                if not util.IsInXLAContext(value.op):
                    max_size = constant_op.constant(-1, dtypes.int32)
                else:
                    max_size = _GetMaxSizeFromNestedMaximumIterations(value, self.forward_context)
                acc = gen_data_flow_ops.stack_v2(max_size=max_size, elem_type=value.dtype.base_dtype, name='f_acc')
            if curr_ctxt:
                curr_ctxt.Exit()
            enter_acc = self.forward_context.AddValue(acc)
            swap_enabled = self.forward_context.swap_memory
            value_ctxt = util.GetOutputContext(value.op)
            if value_ctxt == self.forward_context:
                self.forward_context.Enter()
                push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory=swap_enabled)
                self.forward_context.Exit()
                self.forward_index.op._add_control_input(push.op)
            else:
                if not isinstance(value_ctxt, control_flow_ops.CondContext):
                    raise TypeError('value_ctxt is not a CondContext: %s' % value_ctxt)
                if dead_branch:
                    value_ctxt.outer_context.Enter()
                    push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory=swap_enabled)
                    value_ctxt.outer_context.Exit()
                    push.op._set_control_flow_context(value_ctxt)
                else:
                    value_ctxt.Enter()
                    push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory=swap_enabled)
                    value_ctxt.Exit()
                self.forward_sync._add_control_input(push.op)
            add_op = self.forward_index.op.inputs[0].op
            push.op._add_control_input(add_op)
            return acc