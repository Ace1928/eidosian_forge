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
def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Calculate a streaming confusion matrix.

  Calculates a confusion matrix. For estimation over a stream of data,
  the function creates an  `update_op` operation.

  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).

  Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
  """
    total_cm = metric_variable([num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')
    predictions = math_ops.cast(predictions, dtypes.int64)
    labels = math_ops.cast(labels, dtypes.int64)
    num_classes = math_ops.cast(num_classes, dtypes.int64)
    if predictions.get_shape().ndims > 1:
        predictions = array_ops.reshape(predictions, [-1])
    if labels.get_shape().ndims > 1:
        labels = array_ops.reshape(labels, [-1])
    if weights is not None and weights.get_shape().ndims > 1:
        weights = array_ops.reshape(weights, [-1])
    current_cm = confusion_matrix.confusion_matrix(labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
    update_op = state_ops.assign_add(total_cm, current_cm)
    return (total_cm, update_op)