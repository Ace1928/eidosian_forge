import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
def _parse_grad_debug_op_name(op_name):
    """Parse the name of a debug gradient op.

  Args:
    op_name: the name of the debug gradient op.

  Returns:
    1) The UUID of the GradientsDebugger that created the debug gradient op.
    2) Name of the original tensor whose gradient is debugged by the debug
       gradient op.
  """
    name_items = op_name.split('/')
    assert len(name_items) > 1
    assert name_items[-1].startswith(_GRADIENT_DEBUG_TAG)
    grad_debugger_uuid = name_items[-1][len(_GRADIENT_DEBUG_TAG):]
    if '_' in grad_debugger_uuid:
        grad_debugger_uuid = grad_debugger_uuid[:grad_debugger_uuid.index('_')]
    orig_tensor_slot = int(name_items[-2][name_items[-2].rfind('_') + 1:])
    orig_base_op_name = name_items[-2][:name_items[-2].rfind('_')]
    orig_tensor_name = '/'.join(name_items[:-2] + [orig_base_op_name]) + ':%d' % orig_tensor_slot
    return (grad_debugger_uuid, orig_tensor_name)