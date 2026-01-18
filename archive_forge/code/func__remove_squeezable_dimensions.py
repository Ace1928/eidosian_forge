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
def _remove_squeezable_dimensions(labels, predictions, weights=None, expected_rank_diff=0):
    """Internal version of _remove_squeezable_dimensions which handles weights.

  Squeezes `predictions` and `labels` if their ranks differ from expected by
  exactly 1.
  Squeezes `weights` if its rank is 1 more than the new rank of `predictions`

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    labels: Label values, a `Tensor` whose dimensions match `predictions`.
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    weights: Optional weight `Tensor`. It will be squeezed if it's not scalar,
      and its rank is 1 more than the new rank of `labels`.
    expected_rank_diff: Expected result of `rank(predictions) - rank(labels)`.

  Returns:
    Tuple of `predictions`, `labels` and `weights`, possibly with the last
    dimension squeezed.
  """
    labels, predictions = confusion_matrix.remove_squeezable_dimensions(labels, predictions, expected_rank_diff=expected_rank_diff)
    if weights is not None:
        weights = ops.convert_to_tensor(weights)
        labels_rank = labels.get_shape().ndims
        weights_shape = weights.get_shape()
        weights_rank = weights_shape.ndims
        if labels_rank is not None and weights_rank is not None:
            rank_diff = weights_rank - labels_rank
            if rank_diff == 1:
                weights = array_ops.squeeze(weights, [-1])
            return (labels, predictions, weights)
        rank_diff = array_ops.rank(weights) - array_ops.rank(labels)
        if weights_rank is None or (weights_rank > 0 and weights_shape.dims[-1].is_compatible_with(1)):
            weights = cond.cond(math_ops.equal(1, rank_diff), lambda: array_ops.squeeze(weights, [-1]), lambda: weights)
    return (labels, predictions, weights)