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
def _GetGrad(grads, t, unconnected_gradients):
    """Gets gradient for tensor "t"."""
    op = t.op
    op_grads = grads.get(op)
    if not op_grads:
        if unconnected_gradients == UnconnectedGradients.ZERO:
            return _ZerosLike(t)
        elif unconnected_gradients == UnconnectedGradients.NONE:
            return None
        else:
            raise ValueError(f"Unknown value for unconnected_gradients: '{unconnected_gradients}'")
    t_grad = op_grads[t.value_index]
    if unconnected_gradients == UnconnectedGradients.ZERO and t_grad is None:
        return _ZerosLike(t)
    assert not isinstance(t_grad, list), 'gradients list should have been aggregated by now.'
    return t_grad