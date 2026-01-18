from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.feature_column import base_feature_layer as kfc
from keras.src.saving.legacy.saved_model import json_utils
from tensorflow.python.util.tf_export import keras_export
Returns a dense tensor corresponding to the `feature_columns`.

        Example usage:

        >>> t1 = tf.feature_column.embedding_column(
        ...    tf.feature_column.categorical_column_with_hash_bucket("t1", 2),
        ...    dimension=8)
        >>> t2 = tf.feature_column.numeric_column('t2')
        >>> feature_layer = tf.compat.v1.keras.layers.DenseFeatures([t1, t2])
        >>> features = {"t1": tf.constant(["a", "b"]),
        ...             "t2": tf.constant([1, 2])}
        >>> dense_tensor = feature_layer(features, training=True)

        Args:
          features: A mapping from key to tensors. `FeatureColumn`s look up via
            these keys. For example `numeric_column('price')` will look at
            'price' key in this dict. Values can be a `SparseTensor` or a
            `Tensor` depends on corresponding `FeatureColumn`.
          cols_to_output_tensors: If not `None`, this will be filled with a dict
            mapping feature columns to output tensors created.
          training: Python boolean or None, indicating whether to the layer is
            being run in training mode. This argument is passed to the call
            method of any `FeatureColumn` that takes a `training` argument. For
            example, if a `FeatureColumn` performed dropout, the column could
            expose a `training` argument to control whether the dropout should
            be applied. If `None`, becomes `tf.keras.backend.learning_phase()`.
            Defaults to `None`.


        Returns:
          A `Tensor` which represents input layer of a model. Its shape
          is (batch_size, first_layer_dimension) and its dtype is `float32`.
          first_layer_dimension is determined based on given `feature_columns`.

        Raises:
          ValueError: If features are not a dictionary.
        