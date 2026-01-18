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