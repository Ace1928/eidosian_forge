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
class _SequentialHead(base_head.Head):
    """Interface for the head of a sequential model.

  A sequential head handles input sequences of different lengths to compute the
  output of a model. It requires a sequence mask tensor, to indicate which steps
  of the sequences are padded and ensure proper aggregation for loss and metrics
  computation. It has a `input_sequence_mask_key` property that specifies which
  tensor of the feature dictionary to use as the sequence mask tensor.

  Such a head can for instance be used with `RNNEstimator` for sequential
  predictions.

  Example of usage:
    ```python
    def _my_model_fn(features, labels, mode, params, config=None):
      feature_layer = tf.feature_column.SequenceFeatureLayer(columns)
      input_layer, sequence_length = feature_layer(features)
      sequence_length_mask = tf.sequence_mask(sequence_length)
      rnn_layer = tf.keras.layers.RNN(cell=tf.keras.layers.SimpleRNNCell(units),
                                      return_sequences=True)
      logits = rnn_layer(input_layer, mask=sequence_length_mask)
      features[sequential_head.input_sequence_mask_key] = sequence_length_mask
      return sequential_head.create_estimator_spec(
          features=features,
          labels=labels,
          mode=mode,
          logits=logits,
          optimizer=optimizer)
    ```
  """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def input_sequence_mask_key(self):
        """Key of the sequence mask tensor in the feature dictionary.

    Returns:
      A string.
    """
        raise NotImplementedError('Calling an abstract method.')