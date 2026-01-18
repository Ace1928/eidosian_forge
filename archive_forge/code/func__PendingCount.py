import collections
import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_state
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, func_graphs, xs_set):
    """Initialize the pending count for ops between two lists of Operations.

  'pending_count[op]' indicates the number of backprop inputs
  to this operation.

  Args:
    to_ops: list of Operations.
    from_ops: list of Operations.
    colocate_gradients_with_ops: Python bool.  See docstring of gradients().
    func_graphs: list of FuncGraphs. This method will traverse through
      these functions if they capture from_ops or any reachable ops. This is
      useful if to_ops occur in a function and from_ops are in an outer function
      or graph.
    xs_set: ObjectIdentitySet of Tensors.

  Returns:
    A tuple containing: (1) the subset of to_ops reachable from from_ops by a
    path of zero or more backpropagatable tensors, (2) a mapping from operation
    to the number of backprop inputs to that op, and (3) a ControlFlowState
    object which is not None if the ops between from_ops and to_ops contain
    control flow loops.
  """
    reached_ops = set()
    _MarkReachedOps(from_ops, reached_ops, func_graphs)
    reachable_to_ops = set((op for op in to_ops if op in reached_ops))
    between_ops = set()
    between_op_list = []
    queue = collections.deque()
    queue.extend(to_ops)
    while queue:
        op = queue.popleft()
        if op in reached_ops:
            between_ops.add(op)
            between_op_list.append(op)
            reached_ops.remove(op)
            for inp in _NonEagerInputs(op, xs_set):
                queue.append(inp.op)
    loop_state = control_flow_state.MaybeCreateControlFlowState(between_op_list, between_ops, colocate_gradients_with_ops)
    pending_count = collections.defaultdict(int)
    for op in between_op_list:
        for x in _NonEagerInputs(op, xs_set):
            if x.op in between_ops:
                pending_count[x.op] += 1
    return (reachable_to_ops, pending_count, loop_state)