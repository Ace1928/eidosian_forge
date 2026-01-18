import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
@ops.RegisterGradient('DebugGradientRefIdentity')
def _identify_gradient_grad_ref(op, dy):
    """Gradient function for the DebugIdentity op."""
    return _identify_gradient_grad(op, dy)