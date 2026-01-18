import logging
from typing import List, Optional, Type, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
from ray.rllib.utils.error import (
from ray.rllib.utils.gym import check_old_gym_env
from ray.rllib.utils.numpy import one_hot, one_hot_multidiscrete
from ray.rllib.utils.spaces.space_utils import (
from ray.util import log_once
from ray.util.annotations import PublicAPI
def _get_int_index(self, idx: int, fill=None, neg_indices_left_of_zero=False, one_hot_discrete=False):
    if idx >= 0 or neg_indices_left_of_zero:
        idx = self.lookback + idx
    if neg_indices_left_of_zero and idx < 0:
        idx = len(self) + self.lookback
    try:
        if self.finalized:
            data = tree.map_structure(lambda s: s[idx], self.data)
        else:
            data = self.data[idx]
    except IndexError as e:
        if fill is not None:
            return get_dummy_batch_for_space(self.space, fill_value=fill, batch_size=0, one_hot_discrete=one_hot_discrete)
        else:
            raise e
    if one_hot_discrete:
        data = self._one_hot(data, self.space_struct)
    return data