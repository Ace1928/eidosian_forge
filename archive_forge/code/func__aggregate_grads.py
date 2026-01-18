import functools
import operator
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import imperative_grad
from tensorflow.python.eager import tape
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _aggregate_grads(gradients):
    """Aggregate gradients from multiple sources.

  Args:
    gradients: A list of 'Tensor' or 'IndexedSlices' gradients.

  Returns:
    If 'gradients' only has 'Tensor', returns an aggregated 'Tensor'.
    Otherwise returns an aggregated 'IndexedSlices'.
  """
    assert gradients, 'No gradients to aggregate'
    if len(gradients) == 1:
        return gradients[0]
    if all((isinstance(g, tensor_lib.Tensor) for g in gradients)):
        return gen_math_ops.add_n(gradients)
    else:
        assert all((isinstance(g, (tensor_lib.Tensor, indexed_slices.IndexedSlices)) for g in gradients))
        return backprop_util.AggregateIndexedSlicesGradients(gradients)