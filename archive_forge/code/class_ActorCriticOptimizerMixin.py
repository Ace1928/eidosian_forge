import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.tf_utils import huber_loss, make_tf_callable
from ray.rllib.utils.typing import (
class ActorCriticOptimizerMixin:
    """Mixin class to generate the necessary optimizers for actor-critic algos.

    - Creates global step for counting the number of update operations.
    - Creates separate optimizers for actor, critic, and alpha.
    """

    def __init__(self, config):
        if config['framework'] == 'tf2':
            self.global_step = get_variable(0, tf_name='global_step')
            self._actor_optimizer = tf.keras.optimizers.Adam(learning_rate=config['optimization']['actor_learning_rate'])
            self._critic_optimizer = [tf.keras.optimizers.Adam(learning_rate=config['optimization']['critic_learning_rate'])]
            if config['twin_q']:
                self._critic_optimizer.append(tf.keras.optimizers.Adam(learning_rate=config['optimization']['critic_learning_rate']))
            self._alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=config['optimization']['entropy_learning_rate'])
        else:
            self.global_step = tf1.train.get_or_create_global_step()
            self._actor_optimizer = tf1.train.AdamOptimizer(learning_rate=config['optimization']['actor_learning_rate'])
            self._critic_optimizer = [tf1.train.AdamOptimizer(learning_rate=config['optimization']['critic_learning_rate'])]
            if config['twin_q']:
                self._critic_optimizer.append(tf1.train.AdamOptimizer(learning_rate=config['optimization']['critic_learning_rate']))
            self._alpha_optimizer = tf1.train.AdamOptimizer(learning_rate=config['optimization']['entropy_learning_rate'])