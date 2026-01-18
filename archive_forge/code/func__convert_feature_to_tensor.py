from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def _convert_feature_to_tensor(self, name, value):
    """Casts features to the correct dtype based on their name."""
    if name in [feature_keys.TrainEvalFeatures.TIMES, feature_keys.PredictionFeatures.TIMES]:
        return tf.cast(value, tf.dtypes.int64)
    if name == feature_keys.TrainEvalFeatures.VALUES:
        return tf.cast(value, self.model.dtype)
    if name == feature_keys.PredictionFeatures.STATE_TUPLE:
        return value
    return tf.compat.v1.convert_to_tensor_or_sparse_tensor(value)