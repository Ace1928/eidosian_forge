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
def _sparse_average_precision_at_top_k(labels, predictions_idx):
    """Computes average precision@k of predictions with respect to sparse labels.

  From en.wikipedia.org/wiki/Information_retrieval#Average_precision, formula
  for each row is:

    AveP = sum_{i=1...k} P_{i} * rel_{i} / num_relevant_items

  A "row" is the elements in dimension [D1, ... DN] of `predictions_idx`,
  `labels`, and the result `Tensors`. In the common case, this is [batch_size].
  Each row of the results contains the average precision for that row.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions_idx`.
      Values should be non-negative. Negative values are ignored.
    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.
      Commonly, N=1 and `predictions_idx` has shape [batch size, k]. The final
      dimension must be set and contains the top `k` predicted class indices.
      [D1, ... DN] must match `labels`. Values should be in range
      [0, num_classes).

  Returns:
    `float64` `Tensor` of shape [D1, ... DN], where each value is the average
    precision for that row.

  Raises:
    ValueError: if the last dimension of predictions_idx is not set.
  """
    with ops.name_scope(None, 'average_precision', (predictions_idx, labels)) as scope:
        predictions_idx = math_ops.cast(predictions_idx, dtypes.int64, name='predictions_idx')
        if predictions_idx.get_shape().ndims == 0:
            raise ValueError('The rank of `predictions_idx` must be at least 1.')
        k = predictions_idx.get_shape().as_list()[-1]
        if k is None:
            raise ValueError('The last dimension of predictions_idx must be set. Currently, it is None.')
        labels = _maybe_expand_labels(labels, predictions_idx)
        predictions_idx_per_k = array_ops.expand_dims(predictions_idx, -1, name='predictions_idx_per_k')
        labels_per_k = _expand_and_tile(labels, multiple=k, dim=-1, name='labels_per_k')
        relevant_per_k = _sparse_true_positive_at_k(labels_per_k, predictions_idx_per_k, name='relevant_per_k')
        tp_per_k = math_ops.cumsum(relevant_per_k, axis=-1, name='tp_per_k')
        retrieved_per_k = math_ops.cumsum(array_ops.ones_like(relevant_per_k), axis=-1, name='retrieved_per_k')
        precision_per_k = math_ops.divide(math_ops.cast(tp_per_k, dtypes.float64), math_ops.cast(retrieved_per_k, dtypes.float64), name='precision_per_k')
        relevant_precision_per_k = math_ops.multiply(precision_per_k, math_ops.cast(relevant_per_k, dtypes.float64), name='relevant_precision_per_k')
        precision_sum = math_ops.reduce_sum(relevant_precision_per_k, axis=(-1,), name='precision_sum')
        num_relevant_items = math_ops.cast(_num_relevant(labels, k), dtypes.float64)
        return math_ops.divide(precision_sum, num_relevant_items, name=scope)