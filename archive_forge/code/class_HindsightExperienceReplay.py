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
class HindsightExperienceReplay(PrioritizedReplay):
    """Replay Buffer with Hindsight Experience Replay.

  Hindsight goals are sampled uniformly from subsequent steps in the
  same window (`future` strategy from https://arxiv.org/pdf/1707.01495).
  They are not guaranteed to come from the same episode.

  This buffer is not threadsafe. Make sure you call insert() and sample() from a
  single thread.
  """

    def __init__(self, size, specs, importance_sampling_exponent, compute_reward_fn, unroll_length, substitution_probability, name='HindsightExperienceReplay'):
        super(HindsightExperienceReplay, self).__init__(size, specs, importance_sampling_exponent, name)
        self._compute_reward_fn = compute_reward_fn
        self._unroll_length = unroll_length
        self._substitution_probability = substitution_probability

    @tf.Module.with_name_scope
    def sample(self, num_samples, priority_exp):
        indices, weights, sampled_values = super(HindsightExperienceReplay, self).sample(num_samples, priority_exp)
        observation = sampled_values.env_outputs.observation
        batch_size, time_horizon = observation['achieved_goal'].shape[:2]

        def compute_goal_reward():
            goal_reward = self._compute_reward_fn(achieved_goal=observation['achieved_goal'][:, 1:], desired_goal=observation['desired_goal'][:, :-1])
            return tf.concat(values=[goal_reward[:, :1] * np.nan, goal_reward], axis=1)
        old_goal_reward = compute_goal_reward()
        assert old_goal_reward.shape == observation['achieved_goal'].shape[:-1]
        goal_ind = tf.concat(values=[tf.random.uniform((batch_size, 1), min(t + 1, time_horizon - 1), time_horizon, dtype=tf.int32) for t in range(time_horizon)], axis=1)
        substituted_goal = tf.gather(observation['achieved_goal'], goal_ind, axis=1, batch_dims=1)
        mask = tf.cast(tfp.distributions.Bernoulli(probs=self._substitution_probability * tf.ones(goal_ind.shape)).sample(), observation['desired_goal'].dtype)
        mask *= tf.cast(~sampled_values.env_outputs.done, observation['desired_goal'].dtype)
        mask = mask[..., tf.newaxis]
        observation['desired_goal'] = mask * substituted_goal + (1 - mask) * observation['desired_goal']
        new_goal_reward = compute_goal_reward()
        assert new_goal_reward.shape == observation['achieved_goal'].shape[:-1]
        sampled_values = sampled_values._replace(env_outputs=sampled_values.env_outputs._replace(reward=sampled_values.env_outputs.reward + (new_goal_reward - old_goal_reward) * tf.cast(~sampled_values.env_outputs.done, tf.float32)))
        assert time_horizon >= self._unroll_length + 1
        unroll_begin_ind = tf.random.uniform((batch_size,), 0, time_horizon - self._unroll_length, dtype=tf.int32)
        unroll_inds = unroll_begin_ind[:, tf.newaxis] + tf.math.cumsum(tf.ones((batch_size, self._unroll_length + 1), tf.int32), axis=1, exclusive=True)
        subsampled_values = tf.nest.map_structure(lambda t: tf.gather(t, unroll_inds, axis=1, batch_dims=1), sampled_values)
        if hasattr(sampled_values, 'agent_state'):
            subsampled_values = subsampled_values._replace(agent_state=sampled_values.agent_state)
        return (indices, weights, subsampled_values)