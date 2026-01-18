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
class _ShardedMutableDenseHashTable(object):
    """A sharded version of _MutableDenseHashTable.

  It is designed to be interface compatible with LookupInterface and
  MutableDenseHashTable, with the exception of the export method, which is
  replaced by an export_sharded method.

  The _ShardedMutableDenseHashTable keeps `num_shards` _MutableDenseHashTable
  internally. The shard is computed via the modulo operation on the key.
  """

    def __init__(self, key_dtype, value_dtype, default_value, empty_key, deleted_key, num_shards=1, checkpoint=True, name='ShardedMutableHashTable'):
        self._key_dtype = key_dtype
        self._value_dtype = value_dtype
        with ops.name_scope(name, 'sharded_mutable_hash_table') as scope:
            table_shards = []
            for i in range(num_shards):
                self._table_name = scope
                table_shards.append(_MutableDenseHashTable(key_dtype=key_dtype, value_dtype=value_dtype, default_value=default_value, empty_key=empty_key, deleted_key=deleted_key, checkpoint=checkpoint, name='%s-%d-of-%d' % (name, i + 1, num_shards)))
            self._table_shards = table_shards
            self._value_shape = self._table_shards[0]._value_shape

    @property
    def name(self):
        return self._table_name

    @property
    def _num_shards(self):
        return len(self._table_shards)

    @property
    def table_shards(self):
        return self._table_shards

    def size(self, name=None):
        with ops.name_scope(name, 'sharded_mutable_hash_table_size'):
            sizes = [self._table_shards[i].size() for i in range(self._num_shards)]
            return tf.math.add_n(sizes)

    def _shard_indices(self, keys):
        key_shape = keys.get_shape()
        if key_shape.ndims > 1:
            keys = tf.reshape(tf.slice(keys, [0, 0], [-1, 1]), [-1])
        indices = tf.math.floormod(tf.math.abs(keys), self._num_shards)
        return tf.cast(indices, tf.dtypes.int32)

    def _check_keys(self, keys):
        if keys.get_shape().ndims != 1 and keys.get_shape().ndims != 2:
            raise ValueError('Expected a vector or matrix for keys, got %s.' % keys.get_shape())

    def lookup(self, keys, name=None):
        """Looks up `keys` in a table, outputs the corresponding values."""
        if keys.dtype.base_dtype != self._key_dtype:
            raise TypeError('Signature mismatch. Keys must be dtype %s, got %s.' % (self._key_dtype, keys.dtype))
        self._check_keys(keys)
        num_shards = self._num_shards
        if num_shards == 1:
            return self._table_shards[0].lookup(keys, name=name)
        shard_indices = self._shard_indices(keys)
        key_shards = tf.dynamic_partition(keys, shard_indices, num_shards)
        value_shards = [self._table_shards[i].lookup(key_shards[i], name=name) for i in range(num_shards)]
        num_keys = tf.compat.v1.shape(keys)[0]
        original_indices = tf.range(num_keys)
        partitioned_indices = tf.dynamic_partition(original_indices, shard_indices, num_shards)
        return tf.dynamic_stitch(partitioned_indices, value_shards)

    def insert(self, keys, values, name=None):
        """Inserts `keys` in a table."""
        self._check_keys(keys)
        num_shards = self._num_shards
        if num_shards == 1:
            return self._table_shards[0].insert(keys, values, name=name)
        shard_indices = self._shard_indices(keys)
        key_shards = tf.dynamic_partition(keys, shard_indices, num_shards)
        value_shards = tf.dynamic_partition(values, shard_indices, num_shards)
        return_values = [self._table_shards[i].insert(key_shards[i], value_shards[i], name=name) for i in range(num_shards)]
        return tf.group(*return_values)

    def export_sharded(self, name=None):
        """Returns lists of the keys and values tensors in the sharded table.

    Args:
      name: name of the table.

    Returns:
      A pair of lists with the first list containing the key tensors and the
        second list containing the value tensors from each shard.
    """
        keys_list = []
        values_list = []
        for table_shard in self._table_shards:
            exported_keys, exported_values = table_shard.export(name=name)
            keys_list.append(exported_keys)
            values_list.append(exported_values)
        return (keys_list, values_list)