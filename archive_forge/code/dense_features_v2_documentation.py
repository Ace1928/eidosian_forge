from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v2 as tf
from keras.src.feature_column import base_feature_layer as kfc
from keras.src.feature_column import dense_features
from keras.src.utils import tf_contextlib
from tensorflow.python.util.tf_export import keras_export
Creates a DenseFeatures object.

        Args:
          feature_columns: An iterable containing the FeatureColumns to use as
            inputs to your model. All items should be instances of classes
            derived from `DenseColumn` such as `numeric_column`,
            `embedding_column`, `bucketized_column`, `indicator_column`. If you
            have categorical features, you can wrap them with an
            `embedding_column` or `indicator_column`.
          trainable:  Boolean, whether the layer's variables will be updated via
            gradient descent during training.
          name: Name to give to the DenseFeatures.
          **kwargs: Keyword arguments to construct a layer.

        Raises:
          ValueError: if an item in `feature_columns` is not a `DenseColumn`.
        