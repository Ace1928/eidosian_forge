from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
class _AdaptiveInputAuxiliaryStatistics(collections.namedtuple('_AdaptiveInputAuxiliaryStatistics', ['max_time_seen', 'chunk_count', 'inter_observation_duration_sum', 'example_count', 'overall_feature_sum', 'overall_feature_sum_of_squares'])):
    """Extra statistics used to incrementally update InputStatistics."""

    def __new__(cls, num_features, dtype):
        return super(InputStatisticsFromMiniBatch._AdaptiveInputAuxiliaryStatistics, cls).__new__(cls, max_time_seen=tf.compat.v1.get_variable(name='max_time_seen', initializer=tf.dtypes.int64.min, dtype=tf.dtypes.int64, trainable=False), chunk_count=tf.compat.v1.get_variable(name='chunk_count', initializer=tf.compat.v1.initializers.zeros(), shape=[], dtype=tf.dtypes.int64, trainable=False), inter_observation_duration_sum=tf.compat.v1.get_variable(name='inter_observation_duration_sum', initializer=tf.compat.v1.initializers.zeros(), shape=[], dtype=dtype, trainable=False), example_count=tf.compat.v1.get_variable(name='example_count', shape=[], dtype=tf.dtypes.int64, trainable=False), overall_feature_sum=tf.compat.v1.get_variable(name='overall_feature_sum', shape=[num_features], dtype=dtype, initializer=tf.compat.v1.initializers.zeros(), trainable=False), overall_feature_sum_of_squares=tf.compat.v1.get_variable(name='overall_feature_sum_of_squares', shape=[num_features], dtype=dtype, initializer=tf.compat.v1.initializers.zeros(), trainable=False))