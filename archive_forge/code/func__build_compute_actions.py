import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import ray
import ray.experimental.tf_utils
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy, PolicyState, PolicySpec
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_run_builder import _TFRunBuilder
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _build_compute_actions(self, builder, *, input_dict=None, obs_batch=None, state_batches=None, prev_action_batch=None, prev_reward_batch=None, episodes=None, explore=None, timestep=None):
    explore = explore if explore is not None else self.config['explore']
    timestep = timestep if timestep is not None else self.global_timestep
    self.exploration.before_compute_actions(timestep=timestep, explore=explore, tf_sess=self.get_session())
    builder.add_feed_dict(self.extra_compute_action_feed_dict())
    if hasattr(self, '_input_dict'):
        for key, value in input_dict.items():
            if key in self._input_dict:
                tree.map_structure(lambda k, v: builder.add_feed_dict({k: v}), self._input_dict[key], value)
    else:
        builder.add_feed_dict({self._obs_input: input_dict[SampleBatch.OBS]})
        if SampleBatch.PREV_ACTIONS in input_dict:
            builder.add_feed_dict({self._prev_action_input: input_dict[SampleBatch.PREV_ACTIONS]})
        if SampleBatch.PREV_REWARDS in input_dict:
            builder.add_feed_dict({self._prev_reward_input: input_dict[SampleBatch.PREV_REWARDS]})
        state_batches = []
        i = 0
        while 'state_in_{}'.format(i) in input_dict:
            state_batches.append(input_dict['state_in_{}'.format(i)])
            i += 1
        builder.add_feed_dict(dict(zip(self._state_inputs, state_batches)))
    if 'state_in_0' in input_dict and SampleBatch.SEQ_LENS not in input_dict:
        builder.add_feed_dict({self._seq_lens: np.ones(len(input_dict['state_in_0']))})
    builder.add_feed_dict({self._is_exploring: explore})
    if timestep is not None:
        builder.add_feed_dict({self._timestep: timestep})
    to_fetch = [self._sampled_action] + self._state_outputs + [self.extra_compute_action_fetches()]
    fetches = builder.add_fetches(to_fetch)
    return (fetches[0], fetches[1:-1], fetches[-1])