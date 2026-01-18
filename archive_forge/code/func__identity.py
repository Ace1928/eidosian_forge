import collections
import contextlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _identity(x):
    """Identity op that recognizes `TensorArray`, `Operation`, and `Tensor`."""
    if isinstance(x, tensor_array_ops.TensorArray):
        return x.identity()
    elif isinstance(x, ops.Operation):
        return control_flow_ops.group(x)
    elif context.executing_eagerly() and x is None:
        return None
    else:
        return array_ops.identity(x)