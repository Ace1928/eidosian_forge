from tensorflow.python.debug.lib import debug_gradients  # pylint: disable=unused-import
from tensorflow.python.debug.lib import dumping_callback  # pylint: disable=unused-import
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops  # pylint: disable=unused-import
from tensorflow.python.ops import control_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import image_grad  # pylint: disable=unused-import
from tensorflow.python.ops import linalg_grad  # pylint: disable=unused-import
from tensorflow.python.ops import linalg_ops  # pylint: disable=unused-import
from tensorflow.python.ops import logging_ops  # pylint: disable=unused-import
from tensorflow.python.ops import manip_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import optional_grad  # pylint: disable=unused-import
from tensorflow.python.ops import random_grad  # pylint: disable=unused-import
from tensorflow.python.ops import rnn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import sdca_ops  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_grad  # pylint: disable=unused-import
from tensorflow.python.ops.signal import fft_ops  # pylint: disable=unused-import
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.training import checkpoint_ops  # pylint: disable=unused-import
from tensorflow.python.util.tf_export import tf_export
def _hessian_vector_product(ys, xs, v):
    """Multiply the Hessian of `ys` wrt `xs` by `v`.

  This is an efficient construction that uses a backprop-like approach
  to compute the product between the Hessian and another vector. The
  Hessian is usually too large to be explicitly computed or even
  represented, but this method allows us to at least multiply by it
  for the same big-O cost as backprop.

  Implicit Hessian-vector products are the main practical, scalable way
  of using second derivatives with neural networks. They allow us to
  do things like construct Krylov subspaces and approximate conjugate
  gradient descent.

  Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
  x, v)` will return an expression that evaluates to the same values
  as (A + A.T) `v`.

  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.

  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.

  Raises:
    ValueError: `xs` and `v` have different length.

  """
    length = len(xs)
    if len(v) != length:
        raise ValueError('xs and v must have the same length.')
    grads = gradients(ys, xs)
    assert len(grads) == length
    elemwise_products = [math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem)) for grad_elem, v_elem in zip(grads, v) if grad_elem is not None]
    return gradients(elemwise_products, xs)