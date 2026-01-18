import logging
import os
import threading
from typing import Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.eager_tf_policy import (
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import (
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
@with_lock
def _compute_actions_helper_rl_module_explore(self, input_dict, _ray_trace_ctx=None):
    self._re_trace_counter += 1
    extra_fetches = {}
    input_dict = NestedDict(input_dict)
    fwd_out = self.model.forward_exploration(input_dict)
    fwd_out = self.maybe_remove_time_dimension(fwd_out)
    action_dist = None
    if SampleBatch.ACTION_DIST_INPUTS in fwd_out:
        action_dist_class = self.model.get_exploration_action_dist_cls()
        action_dist = action_dist_class.from_logits(fwd_out[SampleBatch.ACTION_DIST_INPUTS])
    if SampleBatch.ACTIONS in fwd_out:
        actions = fwd_out[SampleBatch.ACTIONS]
    else:
        if action_dist is None:
            raise KeyError(f"Your RLModule's `forward_exploration()` method must return a dictwith either the {SampleBatch.ACTIONS} key or the {SampleBatch.ACTION_DIST_INPUTS} key in it (or both)!")
        actions = action_dist.sample()
    for k, v in fwd_out.items():
        if k not in [SampleBatch.ACTIONS, 'state_out']:
            extra_fetches[k] = v
    if action_dist is not None:
        logp = action_dist.logp(actions)
        extra_fetches[SampleBatch.ACTION_LOGP] = logp
        extra_fetches[SampleBatch.ACTION_PROB] = tf.exp(logp)
    state_out = convert_to_numpy(fwd_out.get('state_out', {}))
    return (actions, state_out, extra_fetches)