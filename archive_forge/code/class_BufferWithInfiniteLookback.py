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
class BufferWithInfiniteLookback:

    def __init__(self, data: Optional[Union[List, np.ndarray]]=None, lookback: int=0, space: Optional[gym.Space]=None):
        self.data = data if data is not None else []
        self.lookback = lookback
        self.finalized = not isinstance(self.data, list)
        self._final_len = None
        self.space = space
        self.space_struct = get_base_struct_from_space(self.space)

    def append(self, item) -> None:
        """Appends the given item to the end of this buffer."""
        if self.finalized:
            raise RuntimeError(f'Cannot `append` to a finalized {type(self).__name__}.')
        self.data.append(item)

    def extend(self, items):
        """Appends all items in `items` to the end of this buffer."""
        if self.finalized:
            raise RuntimeError(f'Cannot `extend` a finalized {type(self).__name__}.')
        for item in items:
            self.append(item)

    def pop(self, index: int=-1):
        """Removes the item at `index` from this buffer."""
        if self.finalized:
            raise RuntimeError(f'Cannot `pop` from a finalized {type(self).__name__}.')
        return self.data.pop(index)

    def finalize(self):
        """Finalizes this buffer by converting internal data lists into numpy arrays.

        Thereby, if the individual items in the list are complex (nested 2)
        """
        if not self.finalized:
            self._final_len = len(self.data) - self.lookback
            self.data = batch(self.data)
            self.finalized = True

    def get(self, indices: Optional[Union[int, slice, List[int]]]=None, neg_indices_left_of_zero: bool=False, fill: Optional[float]=None, one_hot_discrete: bool=False):
        """Returns data, based on the given args, from this buffer.

        Args:
            indices: A single int is interpreted as an index, from which to return the
                individual data stored at this index.
                A list of ints is interpreted as a list of indices from which to gather
                individual data in a batch of size len(indices).
                A slice object is interpreted as a range of data to be returned.
                Thereby, negative indices by default are interpreted as "before the end"
                unless the `neg_indices_left_of_zero=True` option is used, in which case
                negative indices are interpreted as "before ts=0", meaning going back
                into the lookback buffer.
            neg_indices_left_of_zero: If True, negative values in `indices` are
                interpreted as "before ts=0", meaning going back into the lookback
                buffer. For example, an buffer with data [4, 5, 6,  7, 8, 9],
                where [4, 5, 6] is the lookback buffer range (ts=0 item is 7), will
                respond to `get(-1, neg_indices_left_of_zero=True)` with `6` and to
                `get(slice(-2, 1), neg_indices_left_of_zero=True)` with `[5, 6,  7]`.
            fill: An optional float value to use for filling up the returned results at
                the boundaries. This filling only happens if the requested index range's
                start/stop boundaries exceed the buffer's boundaries (including the
                lookback buffer on the left side). This comes in very handy, if users
                don't want to worry about reaching such boundaries and want to zero-pad.
                For example, a buffer with data [10, 11,  12, 13, 14] and lookback
                buffer size of 2 (meaning `10` and `11` are part of the lookback buffer)
                will respond to `get(slice(-7, -2), fill=0.0)`
                with `[0.0, 0.0, 10, 11, 12]`.
            one_hot_discrete: If True, will return one-hot vectors (instead of
                int-values) for those sub-components of a (possibly complex) space
                that are Discrete or MultiDiscrete. Note that if `fill=0` and the
                requested `indices` are out of the range of our data, the returned
                one-hot vectors will actually be zero-hot (all slots zero).
        """
        if fill is not None and self.space is None:
            raise ValueError(f'Cannot use `fill` argument in `{type(self).__name__}.get()` if a gym.Space was NOT provided during construction!')
        if indices is None:
            data = self._get_all_data(one_hot_discrete=one_hot_discrete)
        elif isinstance(indices, slice):
            data = self._get_slice(indices, fill=fill, neg_indices_left_of_zero=neg_indices_left_of_zero, one_hot_discrete=one_hot_discrete)
        elif isinstance(indices, list):
            data = [self._get_int_index(idx, fill=fill, neg_indices_left_of_zero=neg_indices_left_of_zero, one_hot_discrete=one_hot_discrete) for idx in indices]
            if self.finalized:
                data = batch(data)
        else:
            assert isinstance(indices, int)
            data = self._get_int_index(indices, fill=fill, neg_indices_left_of_zero=neg_indices_left_of_zero, one_hot_discrete=one_hot_discrete)
        return data

    def __getitem__(self, item):
        """Support squared bracket syntax, e.g. buffer[:5]."""
        return self.get(item)

    def __len__(self):
        """Return the length of our data, excluding the lookback buffer."""
        if self._final_len is not None:
            assert self.finalized
            return self._final_len
        return len(self.data) - self.lookback

    def _get_all_data(self, one_hot_discrete=False):
        data = self[:]
        if one_hot_discrete:
            data = self._one_hot(data, space_struct=self.space_struct)
        return data

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

    def _one_hot(self, data, space_struct):
        if space_struct is None:
            raise ValueError(f'Cannot `one_hot` data in `{type(self).__name__}` if a gym.Space was NOT provided during construction!')

        def _convert(dat_, space):
            if isinstance(space, gym.spaces.Discrete):
                return one_hot(dat_, depth=space.n)
            elif isinstance(space, gym.spaces.MultiDiscrete):
                return one_hot_multidiscrete(dat_, depths=space.nvec)
            return dat_
        if isinstance(data, list):
            data = [tree.map_structure(_convert, dslice, space_struct) for dslice in data]
        else:
            data = tree.map_structure(_convert, data, space_struct)
        return data