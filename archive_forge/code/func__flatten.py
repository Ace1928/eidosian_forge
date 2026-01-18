from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.head import multi_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _flatten(self, labels, logits, features):
    """Flattens labels, logits, and features tensors.

    Provided tensors need to have at least two dimensions. The two first
    dimensions of the provided tensors are flattened to one single dimension.
    If a tensor is dense, the sequence mask in the features dictionary is used
    to flatten it.

    Note: If indices of a sparse tensor are not sorted, they will be reordered.

    Args:
      labels: `Tensor` or `SparseTensor` to flatten.
      logits: `Tensor` or `SparseTensor` to flatten.
      features: Dictionary of `Tensor` or `SparseTensor` objects to flatten.

    Returns:
      - Dense `Tensor` with flattened labels.
      - Dense `Tensor` with flattened logits.
      - Dictionary of flattened dense `Tensor` objects.

    Raises:
      ValueError: If the sequence mask is not found in `features`.
      ValueError: If one of the provided tensors to flatten has not at least two
        dimensions.
    """
    if self.input_sequence_mask_key not in features:
        raise ValueError('The provided sequence_length_mask key `{}` should be included in the features dictionary, but was not found. Found keys: {}.'.format(self.input_sequence_mask_key, list(features.keys())))
    sequence_mask = features[self.input_sequence_mask_key]
    if sequence_mask.get_shape().ndims != 2:
        raise ValueError('Mask is expected to have two dimensions, got {} instead.'.format(sequence_mask.get_shape().ndims))
    with ops.name_scope('flatten'):
        expected_length = tf.math.reduce_sum(tf.cast(sequence_mask, tf.dtypes.int32))
        flat_logits = _flatten_tensor(logits, sequence_mask, expected_length)
        flat_labels = _flatten_tensor(labels, sequence_mask, expected_length)
        flat_features = {}
        for column in self._feature_columns:
            if column not in features:
                raise ValueError('`{}` column expected in features dictionary.'.format(column))
            flat_features[column] = _flatten_tensor(features[column], sequence_mask, expected_length)
        return (flat_labels, flat_logits, flat_features)