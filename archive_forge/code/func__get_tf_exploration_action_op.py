import functools
import gymnasium as gym
import numpy as np
from typing import Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import (
from ray.rllib.utils.tf_utils import zero_logps_from_actions
def _get_tf_exploration_action_op(self, action_dist, timestep, explore):
    ts = self.last_timestep + 1
    stochastic_actions = tf.cond(pred=tf.convert_to_tensor(ts < self.random_timesteps), true_fn=lambda: self.random_exploration.get_tf_exploration_action_op(action_dist, explore=True)[0], false_fn=lambda: action_dist.sample())
    deterministic_actions = action_dist.deterministic_sample()
    action = tf.cond(tf.constant(explore) if isinstance(explore, bool) else explore, true_fn=lambda: stochastic_actions, false_fn=lambda: deterministic_actions)
    logp = tf.cond(tf.math.logical_and(explore, tf.convert_to_tensor(ts >= self.random_timesteps)), true_fn=lambda: action_dist.sampled_action_logp(), false_fn=functools.partial(zero_logps_from_actions, deterministic_actions))
    if self.framework == 'tf2':
        self.last_timestep.assign_add(1)
        return (action, logp)
    else:
        assign_op = tf1.assign_add(self.last_timestep, 1) if timestep is None else tf1.assign(self.last_timestep, timestep)
        with tf1.control_dependencies([assign_op]):
            return (action, logp)