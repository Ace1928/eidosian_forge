from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def _process_exogenous_features(self, times, features):
    """Create a single vector from exogenous features.

    Args:
      times: A [batch size, window size] vector of times for this batch,
        primarily used to check the shape information of exogenous features.
      features: A dictionary of exogenous features corresponding to the columns
        in self._exogenous_feature_columns. Each value should have a shape
        prefixed by [batch size, window size].

    Returns:
      A Tensor with shape [batch size, window size, exogenous dimension], where
      the size of the exogenous dimension depends on the exogenous feature
      columns passed to the model's constructor.
    Raises:
      ValueError: If an exogenous feature has an unknown rank.
    """
    if self._exogenous_feature_columns:
        exogenous_features_single_batch_dimension = {}
        for name, tensor in features.items():
            if tensor.get_shape().ndims is None:
                raise ValueError('Features with unknown rank are not supported. Got shape {} for feature {}.'.format(tensor.get_shape(), name))
            tensor_shape_dynamic = tf.compat.v1.shape(tensor)
            tensor = tf.reshape(tensor, tf.concat([[tensor_shape_dynamic[0] * tensor_shape_dynamic[1]], tensor_shape_dynamic[2:]], axis=0))
            if tensor.get_shape().ndims == 1 and tensor.dtype != tf.dtypes.string:
                exogenous_features_single_batch_dimension[name] = tensor[:, None]
            else:
                exogenous_features_single_batch_dimension[name] = tensor
        embedded_exogenous_features_single_batch_dimension = tf.compat.v1.feature_column.input_layer(features=exogenous_features_single_batch_dimension, feature_columns=self._exogenous_feature_columns, trainable=True)
        exogenous_regressors = tf.reshape(embedded_exogenous_features_single_batch_dimension, tf.concat([tf.compat.v1.shape(times), tf.compat.v1.shape(embedded_exogenous_features_single_batch_dimension)[1:]], axis=0))
        exogenous_regressors.set_shape(times.get_shape().concatenate(embedded_exogenous_features_single_batch_dimension.get_shape()[1:]))
        exogenous_regressors = tf.cast(exogenous_regressors, dtype=self.dtype)
    else:
        exogenous_regressors = None
    return exogenous_regressors