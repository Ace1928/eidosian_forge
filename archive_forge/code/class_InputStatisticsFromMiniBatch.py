from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
class InputStatisticsFromMiniBatch(object):
    """Generate statistics from mini-batch input."""

    def __init__(self, num_features, dtype, starting_variance_window_size=16):
        """Configure the input statistics object.

    Args:
      num_features: Number of features for the time series
      dtype: The floating point data type to use.
      starting_variance_window_size: The number of datapoints to use when
        computing the mean and variance at the start of the series.
    """
        self._starting_variance_window_size = starting_variance_window_size
        self._num_features = num_features
        self._dtype = dtype

    def initialize_graph(self, features, update_statistics=True):
        """Create any ops needed to provide input statistics.

    Should be called before statistics are requested.

    Args:
      features: A dictionary, the output of a `TimeSeriesInputFn` (with keys
        TrainEvalFeatures.TIMES and TrainEvalFeatures.VALUES).
      update_statistics: Whether `features` should be used to update adaptive
        statistics. Typically True for training and false for evaluation.

    Returns:
      An InputStatistics object composed of Variables, which will be updated
      based on mini-batches of data if requested.
    """
        if TrainEvalFeatures.TIMES in features and TrainEvalFeatures.VALUES in features:
            times = features[TrainEvalFeatures.TIMES]
            values = features[TrainEvalFeatures.VALUES]
        else:
            times = None
            values = None
        with tf.compat.v1.variable_scope('input_statistics', use_resource=True):
            statistics = self._create_variable_statistics_object()
        with tf.compat.v1.variable_scope('input_statistics_auxiliary', use_resource=True):
            auxiliary_variables = self._AdaptiveInputAuxiliaryStatistics(num_features=self._num_features, dtype=self._dtype)
        if update_statistics and times is not None and (values is not None):
            assign_op = self._update_statistics_from_mini_batch(statistics, auxiliary_variables, times, values)
            with tf.control_dependencies([assign_op]):
                stat_variables = tf.nest.pack_sequence_as(statistics, [tf.identity(tensor) for tensor in tf.nest.flatten(statistics)])
                min_time = tf.cast(tf.math.reduce_min(times), tf.dtypes.int64)
                start_time = tf.math.minimum(stat_variables.start_time, min_time)
                return stat_variables._replace(start_time=start_time)
        else:
            return statistics

    class _AdaptiveInputAuxiliaryStatistics(collections.namedtuple('_AdaptiveInputAuxiliaryStatistics', ['max_time_seen', 'chunk_count', 'inter_observation_duration_sum', 'example_count', 'overall_feature_sum', 'overall_feature_sum_of_squares'])):
        """Extra statistics used to incrementally update InputStatistics."""

        def __new__(cls, num_features, dtype):
            return super(InputStatisticsFromMiniBatch._AdaptiveInputAuxiliaryStatistics, cls).__new__(cls, max_time_seen=tf.compat.v1.get_variable(name='max_time_seen', initializer=tf.dtypes.int64.min, dtype=tf.dtypes.int64, trainable=False), chunk_count=tf.compat.v1.get_variable(name='chunk_count', initializer=tf.compat.v1.initializers.zeros(), shape=[], dtype=tf.dtypes.int64, trainable=False), inter_observation_duration_sum=tf.compat.v1.get_variable(name='inter_observation_duration_sum', initializer=tf.compat.v1.initializers.zeros(), shape=[], dtype=dtype, trainable=False), example_count=tf.compat.v1.get_variable(name='example_count', shape=[], dtype=tf.dtypes.int64, trainable=False), overall_feature_sum=tf.compat.v1.get_variable(name='overall_feature_sum', shape=[num_features], dtype=dtype, initializer=tf.compat.v1.initializers.zeros(), trainable=False), overall_feature_sum_of_squares=tf.compat.v1.get_variable(name='overall_feature_sum_of_squares', shape=[num_features], dtype=dtype, initializer=tf.compat.v1.initializers.zeros(), trainable=False))

    def _update_statistics_from_mini_batch(self, statistics, auxiliary_variables, times, values):
        """Given mini-batch input, update `statistics` and `auxiliary_variables`."""
        values = tf.cast(values, self._dtype)
        batch_inter_observation_duration = tf.cast(tf.math.reduce_max(times, axis=1) - tf.math.reduce_min(times, axis=1), self._dtype) / tf.cast(tf.compat.v1.shape(times)[1] - 1, self._dtype)
        with tf.compat.v1.device(auxiliary_variables.max_time_seen.device):
            latest_time = tf.cast(tf.math.reduce_max(times), tf.dtypes.int64)
            max_time_seen = tf.math.maximum(auxiliary_variables.max_time_seen, latest_time)
            max_time_seen_assign = tf.compat.v1.assign(auxiliary_variables.max_time_seen, max_time_seen)
        with tf.compat.v1.device(auxiliary_variables.chunk_count.device):
            chunk_count_assign = tf.compat.v1.assign_add(auxiliary_variables.chunk_count, tf.compat.v1.shape(times, out_type=tf.dtypes.int64)[0])
        with tf.compat.v1.device(auxiliary_variables.inter_observation_duration_sum.device):
            inter_observation_duration_assign = tf.compat.v1.assign_add(auxiliary_variables.inter_observation_duration_sum, tf.math.reduce_sum(batch_inter_observation_duration))
        with tf.compat.v1.device(auxiliary_variables.example_count.device):
            example_count_assign = tf.compat.v1.assign_add(auxiliary_variables.example_count, tf.compat.v1.size(times, out_type=tf.dtypes.int64))
        with tf.compat.v1.device(auxiliary_variables.overall_feature_sum.device):
            overall_feature_sum_assign = tf.compat.v1.assign_add(auxiliary_variables.overall_feature_sum, tf.math.reduce_sum(values, axis=[0, 1]))
        with tf.compat.v1.device(auxiliary_variables.overall_feature_sum_of_squares.device):
            overall_feature_sum_of_squares_assign = tf.compat.v1.assign_add(auxiliary_variables.overall_feature_sum_of_squares, tf.math.reduce_sum(values ** 2, axis=[0, 1]))
        per_chunk_aux_updates = tf.group(max_time_seen_assign, chunk_count_assign, inter_observation_duration_assign, example_count_assign, overall_feature_sum_assign, overall_feature_sum_of_squares_assign)
        with tf.control_dependencies([per_chunk_aux_updates]):
            example_count_float = tf.cast(auxiliary_variables.example_count, self._dtype)
            new_feature_mean = auxiliary_variables.overall_feature_sum / example_count_float
            overall_feature_mean_update = tf.compat.v1.assign(statistics.overall_feature_moments.mean, new_feature_mean)
            overall_feature_var_update = tf.compat.v1.assign(statistics.overall_feature_moments.variance, example_count_float / (example_count_float - 1.0) * (auxiliary_variables.overall_feature_sum_of_squares / example_count_float - new_feature_mean ** 2))
            min_time_batch = tf.cast(tf.compat.v1.math.argmin(times[:, 0]), tf.dtypes.int32)

            def series_start_updates():
                mean, variance = tf.compat.v1.nn.moments(values[min_time_batch, :self._starting_variance_window_size], axes=[0])
                return tf.group(tf.compat.v1.assign(statistics.series_start_moments.mean, mean), tf.compat.v1.assign(statistics.series_start_moments.variance, variance))
            with tf.compat.v1.device(statistics.start_time.device):
                series_start_update = tf.compat.v1.cond(tf.math.less_equal(times[min_time_batch, 0], tf.cast(statistics.start_time, times.dtype)), series_start_updates, tf.no_op)
                with tf.control_dependencies([series_start_update]):
                    min_time = tf.cast(tf.math.reduce_min(times), tf.dtypes.int64)
                    start_time = tf.math.minimum(statistics.start_time, min_time)
                    start_time_update = tf.compat.v1.assign(statistics.start_time, start_time)
            inter_observation_duration_estimate = auxiliary_variables.inter_observation_duration_sum / tf.cast(auxiliary_variables.chunk_count, self._dtype)
            total_observation_count_update = tf.compat.v1.assign(statistics.total_observation_count, tf.cast(gen_math_ops.round(tf.cast(max_time_seen_assign - start_time_update + 1, self._dtype) / inter_observation_duration_estimate), tf.dtypes.int64))
            per_chunk_stat_updates = tf.group(overall_feature_mean_update, overall_feature_var_update, series_start_update, start_time_update, total_observation_count_update)
        return per_chunk_stat_updates

    def _create_variable_statistics_object(self):
        """Creates non-trainable variables representing input statistics."""
        series_start_moments = Moments(mean=tf.compat.v1.get_variable(name='series_start_mean', shape=[self._num_features], dtype=self._dtype, initializer=tf.compat.v1.initializers.zeros(), trainable=False), variance=tf.compat.v1.get_variable(name='series_start_variance', shape=[self._num_features], dtype=self._dtype, initializer=tf.compat.v1.initializers.ones(), trainable=False))
        overall_feature_moments = Moments(mean=tf.compat.v1.get_variable(name='overall_feature_mean', shape=[self._num_features], dtype=self._dtype, initializer=tf.compat.v1.initializers.zeros(), trainable=False), variance=tf.compat.v1.get_variable(name='overall_feature_var', shape=[self._num_features], dtype=self._dtype, initializer=tf.compat.v1.initializers.ones(), trainable=False))
        start_time = tf.compat.v1.get_variable(name='start_time', dtype=tf.dtypes.int64, initializer=tf.dtypes.int64.max, trainable=False)
        total_observation_count = tf.compat.v1.get_variable(name='total_observation_count', shape=[], dtype=tf.dtypes.int64, initializer=tf.compat.v1.initializers.ones(), trainable=False)
        return InputStatistics(series_start_moments=series_start_moments, overall_feature_moments=overall_feature_moments, start_time=start_time, total_observation_count=total_observation_count)