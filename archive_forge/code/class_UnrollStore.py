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
class UnrollStore(tf.Module):
    """Utility module for combining individual environment steps into unrolls."""

    def __init__(self, num_envs, unroll_length, timestep_specs, num_overlapping_steps=0, name='UnrollStore'):
        super(UnrollStore, self).__init__(name=name)
        with self.name_scope:
            self._full_length = num_overlapping_steps + unroll_length + 1

            def create_unroll_variable(spec):
                z = tf.zeros([num_envs, self._full_length] + spec.shape.dims, dtype=spec.dtype)
                return tf.Variable(z, trainable=False, name=spec.name)
            self._unroll_length = unroll_length
            self._num_overlapping_steps = num_overlapping_steps
            self._state = tf.nest.map_structure(create_unroll_variable, timestep_specs)
            self._index = tf.Variable(tf.fill([num_envs], tf.constant(num_overlapping_steps, tf.int32)), trainable=False, name='index')

    @property
    def unroll_specs(self):
        return tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape[1:], v.dtype), self._state)

    @tf.function
    @tf.Module.with_name_scope
    def append(self, env_ids, values):
        """Appends values and returns completed unrolls.

    Args:
      env_ids: 1D tensor with the list of environment IDs for which we append
        data.
        There must not be duplicates.
      values: Values to add for each environment. This is a structure
        (in the tf.nest sense) of tensors following "timestep_specs", with a
        batch front dimension which must be equal to the length of 'env_ids'.

    Returns:
      A pair of:
        - 1D tensor of the environment IDs of the completed unrolls.
        - Completed unrolls. This is a structure of tensors following
          'timestep_specs', with added front dimensions: [num_completed_unrolls,
          num_overlapping_steps + unroll_length + 1].
    """
        tf.debugging.assert_equal(tf.shape(env_ids), tf.shape(tf.unique(env_ids)[0]), message=f'Duplicate environment ids in store {self.name}')
        tf.nest.map_structure(lambda s: tf.debugging.assert_equal(tf.shape(env_ids)[0], tf.shape(s)[0], message=f'Batch dimension must equal the number of environments in store {self.name}.'), values)
        curr_indices = self._index.sparse_read(env_ids)
        unroll_indices = tf.stack([env_ids, curr_indices], axis=-1)
        for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
            s.scatter_nd_update(unroll_indices, v)
        self._index.scatter_add(tf.IndexedSlices(1, env_ids))
        return self._complete_unrolls(env_ids)

    @tf.function
    @tf.Module.with_name_scope
    def reset(self, env_ids):
        """Resets state.

    Note, this is only intended to be called when environments need to be reset
    after preemptions. Not at episode boundaries.

    Args:
      env_ids: The environments that need to have their state reset.
    """
        self._index.scatter_update(tf.IndexedSlices(self._num_overlapping_steps, env_ids))
        j = self._num_overlapping_steps
        repeated_env_ids = tf.reshape(tf.tile(tf.expand_dims(tf.cast(env_ids, tf.int64), -1), [1, j]), [-1])
        repeated_range = tf.tile(tf.range(j, dtype=tf.int64), [tf.shape(env_ids)[0]])
        indices = tf.stack([repeated_env_ids, repeated_range], axis=-1)
        for s in tf.nest.flatten(self._state):
            z = tf.zeros(tf.concat([tf.shape(repeated_env_ids), s.shape[2:]], axis=0), s.dtype)
            s.scatter_nd_update(indices, z)

    def _complete_unrolls(self, env_ids):
        env_indices = self._index.sparse_read(env_ids)
        env_ids = tf.gather(env_ids, tf.where(tf.equal(env_indices, self._full_length))[:, 0])
        env_ids = tf.cast(env_ids, tf.int64)
        unrolls = tf.nest.map_structure(lambda s: s.sparse_read(env_ids), self._state)
        j = self._num_overlapping_steps + 1
        repeated_start_range = tf.tile(tf.range(j, dtype=tf.int64), [tf.shape(env_ids)[0]])
        repeated_end_range = tf.tile(tf.range(self._full_length - j, self._full_length, dtype=tf.int64), [tf.shape(env_ids)[0]])
        repeated_env_ids = tf.reshape(tf.tile(tf.expand_dims(env_ids, -1), [1, j]), [-1])
        start_indices = tf.stack([repeated_env_ids, repeated_start_range], -1)
        end_indices = tf.stack([repeated_env_ids, repeated_end_range], -1)
        for s in tf.nest.flatten(self._state):
            s.scatter_nd_update(start_indices, s.gather_nd(end_indices))
        self._index.scatter_update(tf.IndexedSlices(1 + self._num_overlapping_steps, env_ids))
        return (env_ids, unrolls)