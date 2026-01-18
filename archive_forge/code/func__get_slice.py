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
def _get_slice(self, slice_, fill=None, neg_indices_left_of_zero=False, one_hot_discrete=False):
    len_self_plus_lookback = len(self) + self.lookback
    fill_left_count = fill_right_count = 0
    start = slice_.start
    stop = slice_.stop
    if start is None:
        start = self.lookback
    elif start < 0:
        if neg_indices_left_of_zero:
            start = self.lookback + start
        else:
            start = len_self_plus_lookback + start
    else:
        start = self.lookback + start
    if stop is None:
        stop = len_self_plus_lookback
    elif stop < 0:
        if neg_indices_left_of_zero:
            stop = self.lookback + stop
        else:
            stop = len_self_plus_lookback + stop
    else:
        stop = self.lookback + stop
    if start < 0 and stop < 0:
        fill_left_count = abs(start - stop)
        fill_right_count = 0
        start = stop = 0
    elif start >= len_self_plus_lookback and stop >= len_self_plus_lookback:
        fill_right_count = abs(start - stop)
        fill_left_count = 0
        start = stop = len_self_plus_lookback
    elif start < 0:
        fill_left_count = -start
        start = 0
    elif stop >= len_self_plus_lookback:
        fill_right_count = stop - len_self_plus_lookback
        stop = len_self_plus_lookback
    assert start >= 0 and stop >= 0, (start, stop)
    assert start <= len_self_plus_lookback and stop <= len_self_plus_lookback, (start, stop)
    slice_ = slice(start, stop, slice_.step)
    if self.finalized:
        data_slice = tree.map_structure(lambda s: s[slice_], self.data)
    else:
        data_slice = self.data[slice_]
    if one_hot_discrete:
        data_slice = self._one_hot(data_slice, space_struct=self.space_struct)
    if fill is not None and (fill_right_count > 0 or fill_left_count > 0):
        if self.finalized:
            if fill_left_count:
                fill_batch = get_dummy_batch_for_space(self.space, fill_value=fill, batch_size=fill_left_count, one_hot_discrete=one_hot_discrete)
                data_slice = tree.map_structure(lambda s0, s: np.concatenate([s0, s]), fill_batch, data_slice)
            if fill_right_count:
                fill_batch = get_dummy_batch_for_space(self.space, fill_value=fill, batch_size=fill_right_count, one_hot_discrete=one_hot_discrete)
                data_slice = tree.map_structure(lambda s0, s: np.concatenate([s, s0]), fill_batch, data_slice)
        else:
            fill_batch = [get_dummy_batch_for_space(self.space, fill_value=fill, batch_size=0, one_hot_discrete=one_hot_discrete)]
            data_slice = fill_batch * fill_left_count + data_slice + fill_batch * fill_right_count
    return data_slice