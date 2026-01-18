from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
def _num_present(losses, weights, per_batch=False):
    """Computes the number of elements in the loss function induced by `weights`.

  A given weights tensor induces different numbers of usable elements in the
  `losses` tensor. The `weights` tensor is broadcast across `losses` for all
  possible dimensions. For example, if `losses` is a tensor of dimension
  `[4, 5, 6, 3]` and `weights` is a tensor of shape `[4, 5]`, then `weights` is,
  in effect, tiled to match the shape of `losses`. Following this effective
  tile, the total number of present elements is the number of non-zero weights.

  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: `Tensor` of shape `[]`, `[batch_size]` or
      `[batch_size, d1, ... dK]`, where K < N.
    per_batch: Whether to return the number of elements per batch or as a sum
      total.

  Returns:
    The number of present (non-zero) elements in the losses tensor. If
      `per_batch` is `True`, the value is returned as a tensor of size
      `[batch_size]`. Otherwise, a single scalar tensor is returned.
  """
    if isinstance(weights, float) and weights != 0.0 or (context.executing_eagerly() and weights._rank() == 0 and (not math_ops.equal(weights, 0.0))):
        return _num_elements(losses)
    with ops.name_scope(None, 'num_present', (losses, weights)) as scope:
        weights = math_ops.cast(weights, dtype=dtypes.float32)
        present = array_ops.where(math_ops.equal(weights, 0.0), array_ops.zeros_like(weights), array_ops.ones_like(weights))
        present = weights_broadcast_ops.broadcast_weights(present, losses)
        if per_batch:
            return math_ops.reduce_sum(present, axis=math_ops.range(1, array_ops.rank(present)), keepdims=True, name=scope)
        return math_ops.reduce_sum(present, name=scope)