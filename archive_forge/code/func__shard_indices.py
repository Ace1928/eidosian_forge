from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.checkpoint import saveable_compat
def _shard_indices(self, keys):
    key_shape = keys.get_shape()
    if key_shape.ndims > 1:
        keys = tf.reshape(tf.slice(keys, [0, 0], [-1, 1]), [-1])
    indices = tf.math.floormod(tf.math.abs(keys), self._num_shards)
    return tf.cast(indices, tf.dtypes.int32)