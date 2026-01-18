from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _streaming_sparse_false_negative_at_k(labels, predictions_idx, k, class_id=None, weights=None, name=None):
    """Calculates weighted per step false negatives for recall@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    name: Name of new variable, and namespace for other dependent ops.

  Returns:
    A tuple of `Variable` and update `Operation`.

  Raises:
    ValueError: If `weights` is not `None` and has an incompatible shape.
  """
    with ops.name_scope(name, _at_k_name('false_negative', k, class_id=class_id), (predictions_idx, labels, weights)) as scope:
        fn = _sparse_false_negative_at_k(predictions_idx=predictions_idx, labels=labels, class_id=class_id, weights=weights)
        batch_total_fn = math_ops.cast(math_ops.reduce_sum(fn), dtypes.float64)
        var = metric_variable([], dtypes.float64, name=scope)
        return (var, state_ops.assign_add(var, batch_total_fn, name='update'))