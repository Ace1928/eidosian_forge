from collections import OrderedDict
import gymnasium as gym
import logging
import re
import tree  # pip install dm_tree
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy import TFMultiGPUTowerStack
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import (
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _init_action_fetches(self, timestep: Union[int, TensorType], explore: Union[bool, TensorType]) -> Tuple[TensorType, TensorType, TensorType, type, Dict[str, TensorType]]:
    """Create action related fields for base Policy and loss initialization."""
    sampled_action = None
    sampled_action_logp = None
    dist_inputs = None
    extra_action_fetches = {}
    self._state_out = None
    if not self._is_tower:
        self.exploration = self._create_exploration()
        if is_overridden(self.action_sampler_fn):
            sampled_action, sampled_action_logp, dist_inputs, self._state_out = self.action_sampler_fn(self.model, obs_batch=self._input_dict[SampleBatch.OBS], state_batches=self._state_inputs, seq_lens=self._seq_lens, prev_action_batch=self._input_dict.get(SampleBatch.PREV_ACTIONS), prev_reward_batch=self._input_dict.get(SampleBatch.PREV_REWARDS), explore=explore, is_training=self._input_dict.is_training)
        else:
            if is_overridden(self.action_distribution_fn):
                in_dict = self._input_dict
                dist_inputs, self.dist_class, self._state_out = self.action_distribution_fn(self.model, obs_batch=in_dict[SampleBatch.OBS], state_batches=self._state_inputs, seq_lens=self._seq_lens, explore=explore, timestep=timestep, is_training=in_dict.is_training)
            elif isinstance(self.model, tf.keras.Model):
                dist_inputs, self._state_out, extra_action_fetches = self.model(self._input_dict)
            else:
                dist_inputs, self._state_out = self.model(self._input_dict)
            action_dist = self.dist_class(dist_inputs, self.model)
            sampled_action, sampled_action_logp = self.exploration.get_exploration_action(action_distribution=action_dist, timestep=timestep, explore=explore)
    if dist_inputs is not None:
        extra_action_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs
    if sampled_action_logp is not None:
        extra_action_fetches[SampleBatch.ACTION_LOGP] = sampled_action_logp
        extra_action_fetches[SampleBatch.ACTION_PROB] = tf.exp(tf.cast(sampled_action_logp, tf.float32))
    return (sampled_action, sampled_action_logp, dist_inputs, extra_action_fetches)