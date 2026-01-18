from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from keras.src.saving import serialization_lib
class _BaseFeaturesLayer(Layer):
    """Base class for DenseFeatures and SequenceFeatures.

    Defines common methods and helpers.

    Args:
      feature_columns: An iterable containing the FeatureColumns to use as
        inputs to your model.
      expected_column_type: Expected class for provided feature columns.
      trainable:  Boolean, whether the layer's variables will be updated via
        gradient descent during training.
      name: Name to give to the DenseFeatures.
      **kwargs: Keyword arguments to construct a layer.

    Raises:
      ValueError: if an item in `feature_columns` doesn't match
        `expected_column_type`.
    """

    def __init__(self, feature_columns, expected_column_type, trainable, name, partitioner=None, **kwargs):
        super().__init__(name=name, trainable=trainable, **kwargs)
        self._feature_columns = _normalize_feature_columns(feature_columns)
        self._state_manager = tf.__internal__.feature_column.StateManager(self, self.trainable)
        self._partitioner = partitioner
        for column in self._feature_columns:
            if not isinstance(column, expected_column_type):
                raise ValueError('Items of feature_columns must be a {}. You can wrap a categorical column with an embedding_column or indicator_column. Given: {}'.format(expected_column_type, column))

    def build(self, _):
        for column in self._feature_columns:
            with tf.compat.v1.variable_scope(self.name, partitioner=self._partitioner):
                with tf.compat.v1.variable_scope(_sanitize_column_name_for_variable_scope(column.name)):
                    column.create_state(self._state_manager)
        super().build(None)

    def _output_shape(self, input_shape, num_elements):
        """Computes expected output shape of the dense tensor of the layer.

        Args:
          input_shape: Tensor or array with batch shape.
          num_elements: Size of the last dimension of the output.

        Returns:
          Tuple with output shape.
        """
        raise NotImplementedError('Calling an abstract method.')

    def compute_output_shape(self, input_shape):
        total_elements = 0
        for column in self._feature_columns:
            total_elements += column.variable_shape.num_elements()
        return self._target_shape(input_shape, total_elements)

    def _process_dense_tensor(self, column, tensor):
        """Reshapes the dense tensor output of a column based on expected shape.

        Args:
          column: A DenseColumn or SequenceDenseColumn object.
          tensor: A dense tensor obtained from the same column.

        Returns:
          Reshaped dense tensor.
        """
        num_elements = column.variable_shape.num_elements()
        target_shape = self._target_shape(tf.shape(tensor), num_elements)
        return tf.reshape(tensor, shape=target_shape)

    def _verify_and_concat_tensors(self, output_tensors):
        """Verifies and concatenates the dense output of several columns."""
        _verify_static_batch_size_equality(output_tensors, self._feature_columns)
        return tf.concat(output_tensors, -1)

    def get_config(self):
        column_configs = [tf.__internal__.feature_column.serialize_feature_column(fc) for fc in self._feature_columns]
        config = {'feature_columns': column_configs}
        config['partitioner'] = serialization_lib.serialize_keras_object(self._partitioner)
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config_cp = config.copy()
        columns_by_name = {}
        config_cp['feature_columns'] = [tf.__internal__.feature_column.deserialize_feature_column(c, custom_objects, columns_by_name) for c in config['feature_columns']]
        config_cp['partitioner'] = serialization_lib.deserialize_keras_object(config['partitioner'], custom_objects)
        return cls(**config_cp)