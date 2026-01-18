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
def _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state, xs_set):
    """Update pending count for the inputs of op and enqueue ready ops."""
    for x in _NonEagerInputs(op, xs_set):
        pending_count[x.op] -= 1
        ready = pending_count[x.op] == 0
        if loop_state and (not ready):
            ready = pending_count[x.op] > 0 and control_flow_util.IsLoopSwitch(x.op)
        if ready:
            if control_flow_util.IsLoopExit(x.op):
                grad_state = loop_state.GetGradState(x.op, before=False)
                grad_state.deferred_exits.append(x)
                grad_state.pending_exits_count -= 1
                if grad_state.pending_exits_count == 0:
                    has_not_none_grad = False
                    for y in grad_state.deferred_exits:
                        if _HasAnyNotNoneGrads(grads, y.op):
                            has_not_none_grad = True
                            queue.append(y.op)
                        else:
                            grad_state.unused_exits.append(y)
                    if has_not_none_grad:
                        for y in grad_state.unused_exits:
                            if backprop_util.IsTrainable(y):
                                _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
                            queue.append(y.op)
                    else:
                        for y in grad_state.unused_exits:
                            queue.append(y.op)
            else:
                queue.append(x.op)