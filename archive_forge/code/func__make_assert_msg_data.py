import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _make_assert_msg_data(sym, x, y, summarize, test_op):
    """Subroutine of _binary_assert that generates the components of the default error message when running in eager mode.

  Args:
    sym: Mathematical symbol for the test to apply to pairs of tensor elements,
      i.e. "=="
    x: First input to the assertion after applying `convert_to_tensor()`
    y: Second input to the assertion
    summarize: Value of the "summarize" parameter to the original assert_* call;
      tells how many elements of each tensor to print.
    test_op: TensorFlow op that returns a Boolean tensor with True in each
      position where the assertion is satisfied.

  Returns:
    List of tensors and scalars that, when stringified and concatenated,
    will produce the error message string.
  """
    data = []
    data.append('Condition x %s y did not hold.' % sym)
    if summarize > 0:
        if x.shape == y.shape and x.shape.as_list():
            mask = math_ops.logical_not(test_op)
            indices = array_ops.where(mask)
            indices_np = indices.numpy()
            x_vals = array_ops.boolean_mask(x, mask)
            y_vals = array_ops.boolean_mask(y, mask)
            num_vals = min(summarize, indices_np.shape[0])
            data.append('Indices of first %d different values:' % num_vals)
            data.append(indices_np[:num_vals])
            data.append('Corresponding x values:')
            data.append(x_vals.numpy().reshape((-1,))[:num_vals])
            data.append('Corresponding y values:')
            data.append(y_vals.numpy().reshape((-1,))[:num_vals])
        x_np = x.numpy().reshape((-1,))
        y_np = y.numpy().reshape((-1,))
        x_sum = min(x_np.size, summarize)
        y_sum = min(y_np.size, summarize)
        data.append('First %d elements of x:' % x_sum)
        data.append(x_np[:x_sum])
        data.append('First %d elements of y:' % y_sum)
        data.append(y_np[:y_sum])
    return data