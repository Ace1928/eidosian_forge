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
def _SymGrad(op, out_grads):
    """Backprop through a function call node op given its outputs' gradients."""
    f_in = [x for x in op.inputs] + out_grads
    f_types = [default_gradient.get_zeros_dtype(x) for x in op.inputs]
    f = attr_value_pb2.NameAttrList()
    if _IsPartitionedCall(op):
        f.name = op.get_attr('f').name
    else:
        f.name = op.type
    for k in op.node_def.attr:
        f.attr[k].CopyFrom(op.node_def.attr[k])
    in_grads = gen_functional_ops.symbolic_gradient(input=f_in, Tout=f_types, f=f)
    return in_grads