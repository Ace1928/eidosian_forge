import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import control_flow_util
from keras.src.utils import tf_utils
from tensorflow.python.ops.control_flow_ops import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
def _sync_calculate_mean_and_var(self, x, reduction_axes, keep_dims, mask=None):
    with backend.name_scope('moments'):
        y = tf.cast(x, tf.float32) if x.dtype == tf.float16 else x
        replica_ctx = tf.distribute.get_replica_context()
        if not replica_ctx:
            return self._no_sync_calculate_mean_and_var(x, reduction_axes, keep_dims, mask=mask)
        if mask is not None:
            mask_weights = tf.cast(mask, y.dtype, name='mask_weights')
            mask_weights = tf.expand_dims(mask_weights, axis=-1, name='mask_weights_broadcasted')
            y *= mask_weights
            local_count = tf.broadcast_to(mask_weights, tf.shape(y), name='count')
        else:
            local_count = tf.ones_like(y, name='count')
        local_sum = tf.reduce_sum(y, axis=reduction_axes, keepdims=True)
        local_squared_sum = tf.reduce_sum(tf.square(y), axis=reduction_axes, keepdims=True)
        local_count = tf.reduce_sum(local_count, axis=reduction_axes, keepdims=True)
        y_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_sum)
        y_squared_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_squared_sum)
        count_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_count)
        mean = y_sum / count_sum
        y_squared_mean = y_squared_sum / count_sum
        variance = y_squared_mean - tf.square(mean)
        if not keep_dims:
            mean = tf.squeeze(mean, reduction_axes)
            variance = tf.squeeze(variance, reduction_axes)
        if x.dtype == tf.float16:
            return (tf.cast(mean, tf.float16), tf.cast(variance, tf.float16))
        else:
            return (mean, variance)