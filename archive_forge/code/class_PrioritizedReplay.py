import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  
class PrioritizedReplay(tf.Module):
    """Prioritized Replay Buffer.

  This buffer is not threadsafe. Make sure you call insert() and sample() from a
  single thread.
  """

    def __init__(self, size, specs, importance_sampling_exponent, name='PrioritizedReplay'):
        super(PrioritizedReplay, self).__init__(name=name)
        self._priorities = tf.Variable(tf.zeros([size]), dtype=tf.float32)
        self._buffer = tf.nest.map_structure(lambda ts: tf.Variable(tf.zeros([size] + ts.shape, dtype=ts.dtype)), specs)
        self.num_inserted = tf.Variable(0, dtype=tf.int64)
        self._importance_sampling_exponent = importance_sampling_exponent

    @tf.function
    @tf.Module.with_name_scope
    def insert(self, values, priorities):
        """FIFO insertion/removal.

    Args:
      values: The batched values to insert. The tensors must be of the same
        shape and dtype as the `specs` provided in the constructor, except
        including a batch dimension.
      priorities: <float32>[batch_size] tensor with the priorities of the
        elements we insert.
    Returns:
      The indices of the inserted values.
    """
        tf.nest.assert_same_structure(values, self._buffer)
        values = tf.nest.map_structure(tf.convert_to_tensor, values)
        append_size = tf.nest.flatten(values)[0].shape[0]
        start_index = self.num_inserted
        end_index = start_index + append_size
        size = self._priorities.shape[0]
        insert_indices = tf.range(start_index, end_index) % size
        tf.nest.map_structure(lambda b, v: b.batch_scatter_update(tf.IndexedSlices(v, insert_indices)), self._buffer, values)
        self.num_inserted.assign_add(append_size)
        self._priorities.batch_scatter_update(tf.IndexedSlices(priorities, insert_indices))
        return insert_indices

    @tf.function
    @tf.Module.with_name_scope
    def sample(self, num_samples, priority_exp):
        """Samples items from the replay buffer, using priorities.

    Args:
      num_samples: int, number of replay items to sample.
      priority_exp: Priority exponent. Every item i in the replay buffer will be
        sampled with probability:
         priority[i] ** priority_exp /
             sum(priority[j] ** priority_exp, j \\in [0, num_items))
        Set this to 0 in order to get uniform sampling.

    Returns:
      Tuple of:
        - indices: An int64 tensor of shape [num_samples] with the indices in
          the replay buffer of the sampled items.
        - weights: A float32 tensor of shape [num_samples] with the normalized
          weights of the sampled items.
        - sampled_values: A nested structure following the spec passed in the
          contructor, where each tensor has an added front batch dimension equal
          to 'num_samples'.
    """
        tf.debugging.assert_greater_equal(self.num_inserted, tf.constant(0, tf.int64), message='Cannot sample if replay buffer is empty')
        size = self._priorities.shape[0]
        limit = tf.minimum(tf.cast(size, tf.int64), self.num_inserted)
        if priority_exp == 0:
            indices = tf.random.uniform([num_samples], maxval=limit, dtype=tf.int64)
            weights = tf.ones_like(indices, dtype=tf.float32)
        else:
            prob = self._priorities[:limit] ** priority_exp
            prob /= tf.reduce_sum(prob)
            indices = tf.random.categorical([tf.math.log(prob)], num_samples)[0]
            weights = (1.0 / tf.cast(limit, tf.float32) / tf.gather(prob, indices)) ** self._importance_sampling_exponent
            weights /= tf.reduce_max(weights)
        sampled_values = tf.nest.map_structure(lambda b: b.sparse_read(indices), self._buffer)
        return (indices, weights, sampled_values)

    @tf.function
    @tf.Module.with_name_scope
    def update_priorities(self, indices, priorities):
        """Updates the priorities of the items with the given indices.

    Args:
      indices: <int64>[batch_size] tensor with the indices of the items to
        update. If duplicate indices are provided, the priority that will be set
        among possible ones is not specified.
      priorities: <float32>[batch_size] tensor with the new priorities.
    """
        self._priorities.batch_scatter_update(tf.IndexedSlices(priorities, indices))