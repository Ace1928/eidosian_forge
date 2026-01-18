import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
def _preprocess_grad(grad, body_graph_output, while_op_input, while_op_output):
    """Returns the initial gradient to be used for a given output tensor.

  Args:
    grad: the original gradient Tensor passed to the gradient function.
    body_graph_output: the corresponding Tensor in the body graph.
    while_op_input: the corresponding Tensor input of the While op.
    while_op_output: the corresponding Tensor output of the While op.

  Returns:
    A Tensor or None.
  """
    if not _is_trainable(body_graph_output):
        return None
    if while_op_output.dtype in (dtypes.resource, dtypes.variant) and default_gradient.supports_default_grad(while_op_input) and (grad is None):
        return _zeros_like(while_op_input, while_op_output)
    if isinstance(grad, indexed_slices.IndexedSlices):
        return ops.convert_to_tensor(grad)
    return grad