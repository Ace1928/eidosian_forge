import warnings
import tensorflow as tf
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
from keras.src.backend.config import epsilon
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
def _compute_moments_sync(x, axes, keepdims):
    replica_ctx = tf.distribute.get_replica_context()
    if not replica_ctx:
        return _compute_moments(x, axes, keepdims)
    local_count = tf.ones_like(x, name='count')
    local_sum = tf.reduce_sum(x, axis=axes, keepdims=True)
    local_squared_sum = tf.reduce_sum(tf.square(x), axis=axes, keepdims=True)
    local_count = tf.reduce_sum(local_count, axis=axes, keepdims=True)
    y_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_sum)
    y_squared_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_squared_sum)
    count_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_count)
    mean = tf.math.divide_no_nan(y_sum, count_sum)
    y_squared_mean = tf.math.divide_no_nan(y_squared_sum, count_sum)
    variance = tf.maximum(y_squared_mean - tf.square(mean), 0.0)
    if not keepdims:
        mean = tf.squeeze(mean, axes)
        variance = tf.squeeze(variance, axes)
    return (mean, variance)