from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _check_dense_labels_match_logits_and_reshape(labels, logits, expected_labels_dimension):
    """Checks that labels shape matches logits and reshapes if needed.

  Consider logits of shape [D0, D1, ... DN, logits_dimension]. Then labels
  shape must be [D0, D1, ... DN, expected_labels_dimension].
  If expected_labels_dimension=1, labels could be [D0, D1, ... DN] and this
  method reshapes them to [D0, D1, ... DN, 1].

  Args:
    labels: labels Tensor.
    logits: logits Tensor.
    expected_labels_dimension: Integer.

  Returns:
    Validated and reshaped labels Tensor.
  Raises:
    ValueError: If labels is a SparseTensor.
    ValueError: If labels shape is statically defined and fails validation.
    OpError: If labels shape is not statically defined and fails validation.
  """
    if labels is None:
        raise ValueError('You must provide a labels Tensor. Given: None. Suggested troubleshooting steps: Check that your data contain your label feature. Check that your input_fn properly parses and returns labels.')
    with ops.name_scope(None, 'labels', (labels, logits)) as scope:
        labels = tf.compat.v1.convert_to_tensor_or_sparse_tensor(labels)
        if isinstance(labels, tf.sparse.SparseTensor):
            raise ValueError('SparseTensor labels are not supported. labels must be a Tensor of shape [D0, D1, ..., DN, %s], e.g. [batch_size, %s]. Suggested Fix (1): Check the label feature in your data. Each example must contain %s value(s). If not, your choice of label was probably incorrect. Suggested Fix (2): In your input_fn, use tf.sparse_tensor_to_dense() to turn labels into a Tensor.' % (expected_labels_dimension, expected_labels_dimension, expected_labels_dimension))
        if labels.shape.ndims is not None and logits.shape.ndims is not None and (labels.shape.ndims == logits.shape.ndims - 1):
            labels = tf.compat.v1.expand_dims(labels, -1)
        labels_shape = tf.compat.v1.shape(labels)
        logits_shape = tf.compat.v1.shape(logits)
        err_msg = 'labels shape must be [D0, D1, ... DN, {}]. Suggested Fix: check your n_classes argument to the estimator and/or the shape of your label.'.format(expected_labels_dimension)
        assert_rank = tf.compat.v1.debugging.assert_rank_at_least(labels, 2, message=err_msg)
        with tf.control_dependencies([assert_rank]):
            static_shape = labels.shape
            if static_shape.ndims is not None:
                dim1 = static_shape[-1]
                if dim1 is not None and dim1 != expected_labels_dimension:
                    raise ValueError('Mismatched label shape. Expected labels dimension=%s.  Received %s. Suggested Fix:If your classifier expects one-hot encoding label,check your n_classes argument to the estimator and/or the shape of your label. Otherwise, check the shape of your label.' % (expected_labels_dimension, dim1))
            expected_labels_shape = tf.concat([logits_shape[:-1], [expected_labels_dimension]], axis=0)
            assert_dimension = tf.compat.v1.debugging.assert_equal(expected_labels_shape, labels_shape, message=err_msg, data=['expected_labels_shape: ', expected_labels_shape, 'labels_shape: ', labels_shape])
            with tf.control_dependencies([assert_dimension]):
                return tf.identity(labels, name=scope)