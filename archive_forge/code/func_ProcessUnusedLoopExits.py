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