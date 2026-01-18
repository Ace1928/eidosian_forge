from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head as binary_head_lib
from tensorflow_estimator.python.estimator.head import multi_class_head as multi_head_lib
from tensorflow_estimator.python.estimator.head import sequential_head as seq_head_lib
class RNNModel(tf.keras.models.Model):
    """A Keras RNN model.

  Composition of layers to compute logits from RNN model, along with training
  and inference features. See `tf.keras.models.Model` for more details on Keras
  models.

  Example of usage:

  ```python
  rating = tf.feature_column.embedding_column(
      tf.feature_column.sequence_categorical_column_with_identity('rating', 5),
      10)
  rnn_layer = tf.keras.layers.SimpleRNN(20)
  rnn_model = RNNModel(rnn_layer, units=1, sequence_feature_columns=[rating])

  rnn_model.compile(
      tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
  rnn_model.fit(generator(), epochs=10, steps_per_epoch=100)
  rnn_model.predict({'rating': np.array([[0, 1], [2, 3]])}, steps=1)
  ```
  """

    def __init__(self, rnn_layer, units, sequence_feature_columns, context_feature_columns=None, activation=None, return_sequences=False, **kwargs):
        """Initializes a RNNModel instance.

    Args:
      rnn_layer: A Keras RNN layer.
      units: An int indicating the dimension of the logit layer, and of the
        model output.
      sequence_feature_columns: An iterable containing the `FeatureColumn`s that
        represent sequential input. All items in the set should either be
        sequence columns (e.g. `sequence_numeric_column`) or constructed from
        one (e.g. `embedding_column` with `sequence_categorical_column_*` as
        input).
      context_feature_columns: An iterable containing the `FeatureColumn`s for
        contextual input. The data represented by these columns will be
        replicated and given to the RNN at each timestep. These columns must be
        instances of classes derived from `DenseColumn` such as
        `numeric_column`, not the sequential variants.
      activation: Activation function to apply to the logit layer (for instance
        `tf.keras.activations.sigmoid`). If you don't specify anything, no
        activation is applied.
      return_sequences: A boolean indicating whether to return the last output
        in the output sequence, or the full sequence.
      **kwargs: Additional arguments.

    Raises:
      ValueError: If `units` is not an int.
    """
        super(RNNModel, self).__init__(**kwargs)
        if not isinstance(units, int):
            raise ValueError('units must be an int.  Given type: {}'.format(type(units)))
        self._return_sequences = return_sequences
        self._sequence_feature_columns = sequence_feature_columns
        self._context_feature_columns = context_feature_columns
        self._sequence_features_layer = tf.keras.experimental.SequenceFeatures(sequence_feature_columns)
        self._dense_features_layer = None
        if context_feature_columns:
            self._dense_features_layer = tf.compat.v1.keras.layers.DenseFeatures(context_feature_columns)
        self._rnn_layer = rnn_layer
        self._logits_layer = tf.keras.layers.Dense(units=units, activation=activation, name='logits')

    def call(self, inputs, training=None):
        """Computes the RNN output.

    By default no activation is applied and the logits are returned. To output
    probabilites an activation needs to be specified such as sigmoid or softmax.

    Args:
      inputs: A dict mapping keys to input tensors.
      training: Python boolean indicating whether the layers should behave in
        training mode or in inference mode. This argument is passed to the
        model's layers. This is for instance used with cells that use dropout.

    Returns:
      A `Tensor` with logits from RNN model. It has shape
      (batch_size, time_step, logits_size) if `return_sequence` is `True`,
      (batch_size, logits_size) otherwise.
    """
        if not isinstance(inputs, dict):
            raise ValueError('inputs should be a dictionary of `Tensor`s. Given type: {}'.format(type(inputs)))
        with ops.name_scope('sequence_input_layer'):
            try:
                sequence_input, sequence_length = self._sequence_features_layer(inputs, training=training)
            except TypeError:
                sequence_input, sequence_length = self._sequence_features_layer(inputs)
            tf.compat.v1.summary.histogram('sequence_length', sequence_length)
            if self._context_feature_columns:
                try:
                    context_input = self._dense_features_layer(inputs, training=training)
                except TypeError:
                    context_input = self._dense_features_layer(inputs)
                sequence_input = fc.concatenate_context_input(context_input, sequence_input=sequence_input)
        sequence_length_mask = tf.sequence_mask(sequence_length)
        rnn_outputs = self._rnn_layer(sequence_input, mask=sequence_length_mask, training=training)
        logits = self._logits_layer(rnn_outputs)
        if self._return_sequences:
            logits._keras_mask = sequence_length_mask
        return logits

    def get_config(self):
        """Returns a dictionary with the config of the model."""
        config = {'name': self.name}
        config['rnn_layer'] = {'class_name': self._rnn_layer.__class__.__name__, 'config': self._rnn_layer.get_config()}
        config['units'] = self._logits_layer.units
        config['return_sequences'] = self._return_sequences
        config['activation'] = tf.keras.activations.serialize(self._logits_layer.activation)
        config['sequence_feature_columns'] = fc.serialize_feature_columns(self._sequence_feature_columns)
        config['context_feature_columns'] = fc.serialize_feature_columns(self._context_feature_columns) if self._context_feature_columns else None
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Creates a RNNModel from its config.

    Args:
      config: A Python dictionary, typically the output of `get_config`.
      custom_objects: Optional dictionary mapping names (strings) to custom
        classes or functions to be considered during deserialization.

    Returns:
      A RNNModel.
    """
        rnn_layer = tf.keras.layers.deserialize(config.pop('rnn_layer'), custom_objects=custom_objects)
        sequence_feature_columns = fc.deserialize_feature_columns(config.pop('sequence_feature_columns'), custom_objects=custom_objects)
        context_feature_columns = config.pop('context_feature_columns', None)
        if context_feature_columns:
            context_feature_columns = fc.deserialize_feature_columns(context_feature_columns, custom_objects=custom_objects)
        activation = tf.keras.activations.deserialize(config.pop('activation', None), custom_objects=custom_objects)
        return cls(rnn_layer=rnn_layer, sequence_feature_columns=sequence_feature_columns, context_feature_columns=context_feature_columns, activation=activation, **config)