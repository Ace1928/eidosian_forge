import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
@PublicAPI
class SampleBatch(dict):
    """Wrapper around a dictionary with string keys and array-like values.

    For example, {"obs": [1, 2, 3], "reward": [0, -1, 1]} is a batch of three
    samples, each with an "obs" and "reward" attribute.
    """
    OBS = 'obs'
    NEXT_OBS = 'new_obs'
    ACTIONS = 'actions'
    REWARDS = 'rewards'
    PREV_ACTIONS = 'prev_actions'
    PREV_REWARDS = 'prev_rewards'
    TERMINATEDS = 'terminateds'
    TRUNCATEDS = 'truncateds'
    INFOS = 'infos'
    SEQ_LENS = 'seq_lens'
    T = 't'
    EPS_ID = 'eps_id'
    ENV_ID = 'env_id'
    AGENT_INDEX = 'agent_index'
    UNROLL_ID = 'unroll_id'
    ACTION_DIST_INPUTS = 'action_dist_inputs'
    ACTION_PROB = 'action_prob'
    ACTION_LOGP = 'action_logp'
    ACTION_DIST = 'action_dist'
    VF_PREDS = 'vf_preds'
    VALUES_BOOTSTRAPPED = 'values_bootstrapped'
    OBS_EMBEDS = 'obs_embeds'
    RETURNS_TO_GO = 'returns_to_go'
    ATTENTION_MASKS = 'attention_masks'
    DONES = 'dones'
    CUR_OBS = 'obs'

    @PublicAPI
    def __init__(self, *args, **kwargs):
        """Constructs a sample batch (same params as dict constructor).

        Note: All args and those kwargs not listed below will be passed
        as-is to the parent dict constructor.

        Args:
            _time_major: Whether data in this sample batch
                is time-major. This is False by default and only relevant
                if the data contains sequences.
            _max_seq_len: The max sequence chunk length
                if the data contains sequences.
            _zero_padded: Whether the data in this batch
                contains sequences AND these sequences are right-zero-padded
                according to the `_max_seq_len` setting.
            _is_training: Whether this batch is used for
                training. If False, batch may be used for e.g. action
                computations (inference).
        """
        if SampleBatch.DONES in kwargs:
            raise KeyError('SampleBatch cannot be constructed anymore with a `DONES` key! Instead, set the new TERMINATEDS and TRUNCATEDS keys. The values under DONES will then be automatically computed using terminated|truncated.')
        self.time_major = kwargs.pop('_time_major', None)
        self.max_seq_len = kwargs.pop('_max_seq_len', None)
        self.zero_padded = kwargs.pop('_zero_padded', False)
        self._is_training = kwargs.pop('_is_training', None)
        self.num_grad_updates: Optional[float] = kwargs.pop('_num_grad_updates', None)
        dict.__init__(self, *args, **kwargs)
        self._slice_seq_lens_in_B = False
        self.accessed_keys = set()
        self.added_keys = set()
        self.deleted_keys = set()
        self.intercepted_values = {}
        self.get_interceptor = None
        seq_lens_ = self.get(SampleBatch.SEQ_LENS)
        if seq_lens_ is None or (isinstance(seq_lens_, list) and len(seq_lens_) == 0):
            self.pop(SampleBatch.SEQ_LENS, None)
        elif isinstance(seq_lens_, list):
            self[SampleBatch.SEQ_LENS] = seq_lens_ = np.array(seq_lens_, dtype=np.int32)
        elif torch and torch.is_tensor(seq_lens_) or (tf and tf.is_tensor(seq_lens_)):
            self[SampleBatch.SEQ_LENS] = seq_lens_
        if self.max_seq_len is None and seq_lens_ is not None and (not (tf and tf.is_tensor(seq_lens_))) and (len(seq_lens_) > 0):
            if torch and torch.is_tensor(seq_lens_):
                self.max_seq_len = seq_lens_.max().item()
            else:
                self.max_seq_len = max(seq_lens_)
        if self._is_training is None:
            self._is_training = self.pop('is_training', False)
        for k, v in self.items():
            if isinstance(v, (Number, list)) and (not k == SampleBatch.INFOS):
                self[k] = np.array(v)
        self.count = attempt_count_timesteps(self)
        self._slice_map = []

    @PublicAPI
    def __len__(self) -> int:
        """Returns the amount of samples in the sample batch."""
        return self.count

    @PublicAPI
    def agent_steps(self) -> int:
        """Returns the same as len(self) (number of steps in this batch).

        To make this compatible with `MultiAgentBatch.agent_steps()`.
        """
        return len(self)

    @PublicAPI
    def env_steps(self) -> int:
        """Returns the same as len(self) (number of steps in this batch).

        To make this compatible with `MultiAgentBatch.env_steps()`.
        """
        return len(self)

    @DeveloperAPI
    def enable_slicing_by_batch_id(self):
        self._slice_seq_lens_in_B = True

    @DeveloperAPI
    def disable_slicing_by_batch_id(self):
        self._slice_seq_lens_in_B = False

    @ExperimentalAPI
    def is_terminated_or_truncated(self) -> bool:
        """Returns True if `self` is either terminated or truncated at idx -1."""
        return self[SampleBatch.TERMINATEDS][-1] or (SampleBatch.TRUNCATEDS in self and self[SampleBatch.TRUNCATEDS][-1])

    @ExperimentalAPI
    def is_single_trajectory(self) -> bool:
        """Returns True if this SampleBatch only contains one trajectory.

        This is determined by checking all timesteps (except for the last) for being
        not terminated AND (if applicable) not truncated.
        """
        return not any(self[SampleBatch.TERMINATEDS][:-1]) and (SampleBatch.TRUNCATEDS not in self or not any(self[SampleBatch.TRUNCATEDS][:-1]))

    @staticmethod
    @PublicAPI
    @Deprecated(new='concat_samples() from rllib.policy.sample_batch', error=True)
    def concat_samples(samples):
        pass

    @PublicAPI
    def concat(self, other: 'SampleBatch') -> 'SampleBatch':
        """Concatenates `other` to this one and returns a new SampleBatch.

        Args:
            other: The other SampleBatch object to concat to this one.

        Returns:
            The new SampleBatch, resulting from concating `other` to `self`.

        .. testcode::
            :skipif: True

            import numpy as np
            from ray.rllib.policy.sample_batch import SampleBatch
            b1 = SampleBatch({"a": np.array([1, 2])})
            b2 = SampleBatch({"a": np.array([3, 4, 5])})
            print(b1.concat(b2))

        .. testoutput::

            {"a": np.array([1, 2, 3, 4, 5])}
        """
        return concat_samples([self, other])

    @PublicAPI
    def copy(self, shallow: bool=False) -> 'SampleBatch':
        """Creates a deep or shallow copy of this SampleBatch and returns it.

        Args:
            shallow: Whether the copying should be done shallowly.

        Returns:
            A deep or shallow copy of this SampleBatch object.
        """
        copy_ = {k: v for k, v in self.items()}
        data = tree.map_structure(lambda v: np.array(v, copy=not shallow) if isinstance(v, np.ndarray) else v, copy_)
        copy_ = SampleBatch(data, _time_major=self.time_major, _zero_padded=self.zero_padded, _max_seq_len=self.max_seq_len, _num_grad_updates=self.num_grad_updates)
        copy_.set_get_interceptor(self.get_interceptor)
        copy_.added_keys = self.added_keys
        copy_.deleted_keys = self.deleted_keys
        copy_.accessed_keys = self.accessed_keys
        return copy_

    @PublicAPI
    def rows(self) -> Iterator[Dict[str, TensorType]]:
        """Returns an iterator over data rows, i.e. dicts with column values.

        Note that if `seq_lens` is set in self, we set it to 1 in the rows.

        Yields:
            The column values of the row in this iteration.

        .. testcode::
            :skipif: True

            from ray.rllib.policy.sample_batch import SampleBatch
            batch = SampleBatch({
               "a": [1, 2, 3],
               "b": [4, 5, 6],
               "seq_lens": [1, 2]
            })
            for row in batch.rows():
                print(row)

        .. testoutput::

            {"a": 1, "b": 4, "seq_lens": 1}
            {"a": 2, "b": 5, "seq_lens": 1}
            {"a": 3, "b": 6, "seq_lens": 1}
        """
        seq_lens = None if self.get(SampleBatch.SEQ_LENS, 1) is None else 1
        self_as_dict = {k: v for k, v in self.items()}
        for i in range(self.count):
            yield tree.map_structure_with_path(lambda p, v: v[i] if p[0] != self.SEQ_LENS else seq_lens, self_as_dict)

    @PublicAPI
    def columns(self, keys: List[str]) -> List[any]:
        """Returns a list of the batch-data in the specified columns.

        Args:
            keys: List of column names fo which to return the data.

        Returns:
            The list of data items ordered by the order of column
            names in `keys`.

        .. testcode::
            :skipif: True

            from ray.rllib.policy.sample_batch import SampleBatch
            batch = SampleBatch({"a": [1], "b": [2], "c": [3]})
            print(batch.columns(["a", "b"]))

        .. testoutput::

            [[1], [2]]
        """
        out = []
        for k in keys:
            out.append(self[k])
        return out

    @PublicAPI
    def shuffle(self) -> 'SampleBatch':
        """Shuffles the rows of this batch in-place.

        Returns:
            This very (now shuffled) SampleBatch.

        Raises:
            ValueError: If self[SampleBatch.SEQ_LENS] is defined.

        .. testcode::
            :skipif: True

            from ray.rllib.policy.sample_batch import SampleBatch
            batch = SampleBatch({"a": [1, 2, 3, 4]})
            print(batch.shuffle())

        .. testoutput::

            {"a": [4, 1, 3, 2]}
        """
        if self.get(SampleBatch.SEQ_LENS) is not None:
            raise ValueError('SampleBatch.shuffle not possible when your data has `seq_lens` defined!')
        permutation = np.random.permutation(self.count)
        self_as_dict = {k: v for k, v in self.items()}
        shuffled = tree.map_structure(lambda v: v[permutation], self_as_dict)
        self.update(shuffled)
        self.intercepted_values = {}
        return self

    @PublicAPI
    def split_by_episode(self, key: Optional[str]=None) -> List['SampleBatch']:
        """Splits by `eps_id` column and returns list of new batches.
        If `eps_id` is not present, splits by `dones` instead.

        Args:
            key: If specified, overwrite default and use key to split.

        Returns:
            List of batches, one per distinct episode.

        Raises:
            KeyError: If the `eps_id` AND `dones` columns are not present.

        .. testcode::
            :skipif: True

            from ray.rllib.policy.sample_batch import SampleBatch
            # "eps_id" is present
            batch = SampleBatch(
                {"a": [1, 2, 3], "eps_id": [0, 0, 1]})
            print(batch.split_by_episode())

            # "eps_id" not present, split by "dones" instead
            batch = SampleBatch(
                {"a": [1, 2, 3, 4, 5], "dones": [0, 0, 1, 0, 1]})
            print(batch.split_by_episode())

            # The last episode is appended even if it does not end with done
            batch = SampleBatch(
                {"a": [1, 2, 3, 4, 5], "dones": [0, 0, 1, 0, 0]})
            print(batch.split_by_episode())

            batch = SampleBatch(
                {"a": [1, 2, 3, 4, 5], "dones": [0, 0, 0, 0, 0]})
            print(batch.split_by_episode())


        .. testoutput::

            [{"a": [1, 2], "eps_id": [0, 0]}, {"a": [3], "eps_id": [1]}]
            [{"a": [1, 2, 3], "dones": [0, 0, 1]}, {"a": [4, 5], "dones": [0, 1]}]
            [{"a": [1, 2, 3], "dones": [0, 0, 1]}, {"a": [4, 5], "dones": [0, 0]}]
            [{"a": [1, 2, 3, 4, 5], "dones": [0, 0, 0, 0, 0]}]


        """
        assert key is None or key in [SampleBatch.EPS_ID, SampleBatch.DONES], f"`SampleBatch.split_by_episode(key={key})` invalid! Must be [None|'dones'|'eps_id']."

        def slice_by_eps_id():
            slices = []
            cur_eps_id = self[SampleBatch.EPS_ID][0]
            offset = 0
            for i in range(self.count):
                next_eps_id = self[SampleBatch.EPS_ID][i]
                if next_eps_id != cur_eps_id:
                    slices.append(self[offset:i])
                    offset = i
                    cur_eps_id = next_eps_id
            slices.append(self[offset:self.count])
            return slices

        def slice_by_terminateds_or_truncateds():
            slices = []
            offset = 0
            for i in range(self.count):
                if self[SampleBatch.TERMINATEDS][i] or (SampleBatch.TRUNCATEDS in self and self[SampleBatch.TRUNCATEDS][i]):
                    slices.append(self[offset:i + 1])
                    offset = i + 1
            if offset != self.count:
                slices.append(self[offset:])
            return slices
        key_to_method = {SampleBatch.EPS_ID: slice_by_eps_id, SampleBatch.DONES: slice_by_terminateds_or_truncateds}
        key_resolve_order = [SampleBatch.EPS_ID, SampleBatch.DONES]
        slices = None
        if key is not None:
            if key == SampleBatch.EPS_ID and key not in self:
                raise KeyError(f'{self} does not have key `{key}`!')
            slices = key_to_method[key]()
        else:
            for key in key_resolve_order:
                if key == SampleBatch.DONES or key in self:
                    slices = key_to_method[key]()
                    break
            if slices is None:
                raise KeyError(f'{self} does not have keys {key_resolve_order}!')
        assert sum((s.count for s in slices)) == self.count, f'Calling split_by_episode on {self} returns {slices}'
        f'which should in total have {self.count} timesteps!'
        return slices

    def slice(self, start: int, end: int, state_start=None, state_end=None) -> 'SampleBatch':
        """Returns a slice of the row data of this batch (w/o copying).

        Args:
            start: Starting index. If < 0, will left-zero-pad.
            end: Ending index.

        Returns:
            A new SampleBatch, which has a slice of this batch's data.
        """
        if self.get(SampleBatch.SEQ_LENS) is not None and len(self[SampleBatch.SEQ_LENS]) > 0:
            if start < 0:
                data = {k: np.concatenate([np.zeros(shape=(-start,) + v.shape[1:], dtype=v.dtype), v[0:end]]) for k, v in self.items() if k != SampleBatch.SEQ_LENS and (not k.startswith('state_in_'))}
            else:
                data = {k: tree.map_structure(lambda s: s[start:end], v) for k, v in self.items() if k != SampleBatch.SEQ_LENS and (not k.startswith('state_in_'))}
            if state_start is not None:
                assert state_end is not None
                state_idx = 0
                state_key = 'state_in_{}'.format(state_idx)
                while state_key in self:
                    data[state_key] = self[state_key][state_start:state_end]
                    state_idx += 1
                    state_key = 'state_in_{}'.format(state_idx)
                seq_lens = list(self[SampleBatch.SEQ_LENS][state_start:state_end])
                data_len = len(data[next(iter(data))])
                if sum(seq_lens) != data_len:
                    assert sum(seq_lens) > data_len
                    seq_lens[-1] = data_len - sum(seq_lens[:-1])
            else:
                count = 0
                state_start = None
                seq_lens = None
                for i, seq_len in enumerate(self[SampleBatch.SEQ_LENS]):
                    count += seq_len
                    if count >= end:
                        state_idx = 0
                        state_key = 'state_in_{}'.format(state_idx)
                        if state_start is None:
                            state_start = i
                        while state_key in self:
                            data[state_key] = self[state_key][state_start:i + 1]
                            state_idx += 1
                            state_key = 'state_in_{}'.format(state_idx)
                        seq_lens = list(self[SampleBatch.SEQ_LENS][state_start:i]) + [seq_len - (count - end)]
                        if start < 0:
                            seq_lens[0] += -start
                        diff = sum(seq_lens) - (end - start)
                        if diff > 0:
                            seq_lens[0] -= diff
                        assert sum(seq_lens) == end - start
                        break
                    elif state_start is None and count > start:
                        state_start = i
            return SampleBatch(data, seq_lens=seq_lens, _is_training=self.is_training, _time_major=self.time_major, _num_grad_updates=self.num_grad_updates)
        else:
            return SampleBatch(tree.map_structure(lambda value: value[start:end], self), _is_training=self.is_training, _time_major=self.time_major, _num_grad_updates=self.num_grad_updates)

    def _batch_slice(self, slice_: slice) -> 'SampleBatch':
        """Helper method to handle SampleBatch slicing using a slice object.

        The returned SampleBatch uses the same underlying data object as
        `self`, so changing the slice will also change `self`.

        Note that only zero or positive bounds are allowed for both start
        and stop values. The slice step must be 1 (or None, which is the
        same).

        Args:
            slice_: The python slice object to slice by.

        Returns:
            A new SampleBatch, however "linking" into the same data
            (sliced) as self.
        """
        start = slice_.start or 0
        stop = slice_.stop or len(self[SampleBatch.SEQ_LENS])
        if stop > len(self):
            stop = len(self)
        assert start >= 0 and stop >= 0 and (slice_.step in [1, None])
        data = tree.map_structure(lambda value: value[start:stop], self)
        return SampleBatch(data, _is_training=self.is_training, _time_major=self.time_major, _num_grad_updates=self.num_grad_updates)

    @PublicAPI
    def timeslices(self, size: Optional[int]=None, num_slices: Optional[int]=None, k: Optional[int]=None) -> List['SampleBatch']:
        """Returns SampleBatches, each one representing a k-slice of this one.

        Will start from timestep 0 and produce slices of size=k.

        Args:
            size: The size (in timesteps) of each returned SampleBatch.
            num_slices: The number of slices to produce.
            k: Deprecated: Use size or num_slices instead. The size
                (in timesteps) of each returned SampleBatch.

        Returns:
            The list of `num_slices` (new) SampleBatches or n (new)
            SampleBatches each one of size `size`.
        """
        if size is None and num_slices is None:
            deprecation_warning('k', 'size or num_slices')
            assert k is not None
            size = k
        if size is None:
            assert isinstance(num_slices, int)
            slices = []
            left = len(self)
            start = 0
            while left:
                len_ = left // (num_slices - len(slices))
                stop = start + len_
                slices.append(self[start:stop])
                left -= len_
                start = stop
            return slices
        else:
            assert isinstance(size, int)
            slices = []
            left = len(self)
            start = 0
            while left:
                stop = start + size
                slices.append(self[start:stop])
                left -= size
                start = stop
            return slices

    @Deprecated(new='SampleBatch.right_zero_pad', error=True)
    def zero_pad(self, max_seq_len, exclude_states=True):
        pass

    def right_zero_pad(self, max_seq_len: int, exclude_states: bool=True):
        """Right (adding zeros at end) zero-pads this SampleBatch in-place.

        This will set the `self.zero_padded` flag to True and
        `self.max_seq_len` to the given `max_seq_len` value.

        Args:
            max_seq_len: The max (total) length to zero pad to.
            exclude_states: If False, also right-zero-pad all
                `state_in_x` data. If True, leave `state_in_x` keys
                as-is.

        Returns:
            This very (now right-zero-padded) SampleBatch.

        Raises:
            ValueError: If self[SampleBatch.SEQ_LENS] is None (not defined).

        .. testcode::
            :skipif: True

            from ray.rllib.policy.sample_batch import SampleBatch
            batch = SampleBatch(
                {"a": [1, 2, 3], "seq_lens": [1, 2]})
            print(batch.right_zero_pad(max_seq_len=4))

            batch = SampleBatch({"a": [1, 2, 3],
                                 "state_in_0": [1.0, 3.0],
                                 "seq_lens": [1, 2]})
            print(batch.right_zero_pad(max_seq_len=5))

        .. testoutput::

            {"a": [1, 0, 0, 0, 2, 3, 0, 0], "seq_lens": [1, 2]}
            {"a": [1, 0, 0, 0, 0, 2, 3, 0, 0, 0],
             "state_in_0": [1.0, 3.0],  # <- all state-ins remain as-is
             "seq_lens": [1, 2]}

        """
        seq_lens = self.get(SampleBatch.SEQ_LENS)
        if seq_lens is None:
            raise ValueError(f'Cannot right-zero-pad SampleBatch if no `seq_lens` field present! SampleBatch={self}')
        length = len(seq_lens) * max_seq_len

        def _zero_pad_in_place(path, value):
            if exclude_states is True and path[0].startswith('state_in_') or path[0] == SampleBatch.SEQ_LENS:
                return
            if value.dtype == object or value.dtype.type is np.str_:
                f_pad = [None] * length
            else:
                f_pad = np.zeros((length,) + np.shape(value)[1:], dtype=value.dtype)
            f_pad_base = f_base = 0
            for len_ in self[SampleBatch.SEQ_LENS]:
                f_pad[f_pad_base:f_pad_base + len_] = value[f_base:f_base + len_]
                f_pad_base += max_seq_len
                f_base += len_
            assert f_base == len(value), value
            curr = self
            for i, p in enumerate(path):
                if i == len(path) - 1:
                    curr[p] = f_pad
                curr = curr[p]
        self_as_dict = {k: v for k, v in self.items()}
        tree.map_structure_with_path(_zero_pad_in_place, self_as_dict)
        self.zero_padded = True
        self.max_seq_len = max_seq_len
        return self

    @ExperimentalAPI
    def to_device(self, device, framework='torch'):
        """TODO: transfer batch to given device as framework tensor."""
        if framework == 'torch':
            assert torch is not None
            for k, v in self.items():
                self[k] = convert_to_torch_tensor(v, device)
        else:
            raise NotImplementedError
        return self

    @PublicAPI
    def size_bytes(self) -> int:
        """Returns sum over number of bytes of all data buffers.

        For numpy arrays, we use ``.nbytes``. For all other value types, we use
        sys.getsizeof(...).

        Returns:
            The overall size in bytes of the data buffer (all columns).
        """
        return sum((v.nbytes if isinstance(v, np.ndarray) else sys.getsizeof(v) for v in tree.flatten(self)))

    def get(self, key, default=None):
        """Returns one column (by key) from the data or a default value."""
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    @PublicAPI
    def as_multi_agent(self) -> 'MultiAgentBatch':
        """Returns the respective MultiAgentBatch using DEFAULT_POLICY_ID.

        Returns:
            The MultiAgentBatch (using DEFAULT_POLICY_ID) corresponding
            to this SampleBatch.
        """
        return MultiAgentBatch({DEFAULT_POLICY_ID: self}, self.count)

    @PublicAPI
    def __getitem__(self, key: Union[str, slice]) -> TensorType:
        """Returns one column (by key) from the data or a sliced new batch.

        Args:
            key: The key (column name) to return or
                a slice object for slicing this SampleBatch.

        Returns:
            The data under the given key or a sliced version of this batch.
        """
        if isinstance(key, slice):
            return self._slice(key)
        if key == SampleBatch.DONES:
            return self[SampleBatch.TERMINATEDS]
        elif key == 'is_training':
            if log_once("SampleBatch['is_training']"):
                deprecation_warning(old="SampleBatch['is_training']", new='SampleBatch.is_training', error=False)
            return self.is_training
        if not hasattr(self, key) and key in self:
            self.accessed_keys.add(key)
        value = dict.__getitem__(self, key)
        if self.get_interceptor is not None:
            if key not in self.intercepted_values:
                self.intercepted_values[key] = self.get_interceptor(value)
            value = self.intercepted_values[key]
        return value

    @PublicAPI
    def __setitem__(self, key, item) -> None:
        """Inserts (overrides) an entire column (by key) in the data buffer.

        Args:
            key: The column name to set a value for.
            item: The data to insert.
        """
        if key == SampleBatch.DONES:
            raise KeyError('Cannot set `DONES` anymore in a SampleBatch! Instead, set the new TERMINATEDS and TRUNCATEDS keys. The values under DONES will then be automatically computed using terminated|truncated.')
        elif not hasattr(self, 'added_keys'):
            dict.__setitem__(self, key, item)
            return
        if key == 'is_training':
            if log_once("SampleBatch['is_training']"):
                deprecation_warning(old="SampleBatch['is_training']", new='SampleBatch.is_training', error=False)
            self._is_training = item
            return
        if key not in self:
            self.added_keys.add(key)
        dict.__setitem__(self, key, item)
        if key in self.intercepted_values:
            self.intercepted_values[key] = item

    @property
    def is_training(self):
        if self.get_interceptor is not None and isinstance(self._is_training, bool):
            if '_is_training' not in self.intercepted_values:
                self.intercepted_values['_is_training'] = self.get_interceptor(self._is_training)
            return self.intercepted_values['_is_training']
        return self._is_training

    def set_training(self, training: Union[bool, 'tf1.placeholder']=True):
        """Sets the `is_training` flag for this SampleBatch."""
        self._is_training = training
        self.intercepted_values.pop('_is_training', None)

    @PublicAPI
    def __delitem__(self, key):
        self.deleted_keys.add(key)
        dict.__delitem__(self, key)

    @DeveloperAPI
    def compress(self, bulk: bool=False, columns: Set[str]=frozenset(['obs', 'new_obs'])) -> 'SampleBatch':
        """Compresses the data buffers (by column) in place.

        Args:
            bulk: Whether to compress across the batch dimension (0)
                as well. If False will compress n separate list items, where n
                is the batch size.
            columns: The columns to compress. Default: Only
                compress the obs and new_obs columns.

        Returns:
            This very (now compressed) SampleBatch.
        """

        def _compress_in_place(path, value):
            if path[0] not in columns:
                return
            curr = self
            for i, p in enumerate(path):
                if i == len(path) - 1:
                    if bulk:
                        curr[p] = pack(value)
                    else:
                        curr[p] = np.array([pack(o) for o in value])
                curr = curr[p]
        tree.map_structure_with_path(_compress_in_place, self)
        return self

    @DeveloperAPI
    def decompress_if_needed(self, columns: Set[str]=frozenset(['obs', 'new_obs'])) -> 'SampleBatch':
        """Decompresses data buffers (per column if not compressed) in place.

        Args:
            columns: The columns to decompress. Default: Only
                decompress the obs and new_obs columns.

        Returns:
            This very (now uncompressed) SampleBatch.
        """

        def _decompress_in_place(path, value):
            if path[0] not in columns:
                return
            curr = self
            for p in path[:-1]:
                curr = curr[p]
            if is_compressed(value):
                curr[path[-1]] = unpack(value)
            elif len(value) > 0 and is_compressed(value[0]):
                curr[path[-1]] = np.array([unpack(o) for o in value])
        tree.map_structure_with_path(_decompress_in_place, self)
        return self

    @DeveloperAPI
    def set_get_interceptor(self, fn):
        """Sets a function to be called on every getitem."""
        if fn is not self.get_interceptor:
            self.intercepted_values = {}
        self.get_interceptor = fn

    def __repr__(self):
        keys = list(self.keys())
        if self.get(SampleBatch.SEQ_LENS) is None:
            return f'SampleBatch({self.count}: {keys})'
        else:
            keys.remove(SampleBatch.SEQ_LENS)
            return f'SampleBatch({self.count} (seqs={len(self['seq_lens'])}): {keys})'

    def _slice(self, slice_: slice) -> 'SampleBatch':
        """Helper method to handle SampleBatch slicing using a slice object.

        The returned SampleBatch uses the same underlying data object as
        `self`, so changing the slice will also change `self`.

        Note that only zero or positive bounds are allowed for both start
        and stop values. The slice step must be 1 (or None, which is the
        same).

        Args:
            slice_: The python slice object to slice by.

        Returns:
            A new SampleBatch, however "linking" into the same data
            (sliced) as self.
        """
        if self._slice_seq_lens_in_B:
            return self._batch_slice(slice_)
        start = slice_.start or 0
        stop = slice_.stop or len(self)
        if stop > len(self):
            stop = len(self)
        if self.get(SampleBatch.SEQ_LENS) is not None and len(self[SampleBatch.SEQ_LENS]) > 0:
            if not self._slice_map:
                sum_ = 0
                for i, l in enumerate(map(int, self[SampleBatch.SEQ_LENS])):
                    self._slice_map.extend([(i, sum_)] * l)
                    sum_ = sum_ + l
                self._slice_map.append((len(self[SampleBatch.SEQ_LENS]), sum_))
            start_seq_len, start_unpadded = self._slice_map[start]
            stop_seq_len, stop_unpadded = self._slice_map[stop]
            start_padded = start_unpadded
            stop_padded = stop_unpadded
            if self.zero_padded:
                start_padded = start_seq_len * self.max_seq_len
                stop_padded = stop_seq_len * self.max_seq_len

            def map_(path, value):
                if path[0] != SampleBatch.SEQ_LENS and (not path[0].startswith('state_in_')):
                    if path[0] != SampleBatch.INFOS:
                        return value[start_padded:stop_padded]
                    elif isinstance(value, np.ndarray) and value.size > 0 or (torch and torch.is_tensor(value) and (len(list(value.shape)) > 0)) or (tf and tf.is_tensor(value) and (tf.size(value) > 0)):
                        return value[start_unpadded:stop_unpadded]
                    else:
                        return value
                else:
                    return value[start_seq_len:stop_seq_len]
            data = tree.map_structure_with_path(map_, self)
            if isinstance(data.get(SampleBatch.INFOS), list):
                data[SampleBatch.INFOS] = data[SampleBatch.INFOS][start_unpadded:stop_unpadded]
            return SampleBatch(data, _is_training=self.is_training, _time_major=self.time_major, _zero_padded=self.zero_padded, _max_seq_len=self.max_seq_len if self.zero_padded else None, _num_grad_updates=self.num_grad_updates)
        else:

            def map_(value):
                if isinstance(value, np.ndarray) or (torch and torch.is_tensor(value)) or (tf and tf.is_tensor(value)):
                    return value[start:stop]
                else:
                    return value
            data = tree.map_structure(map_, self)
            return SampleBatch(data, _is_training=self.is_training, _time_major=self.time_major, _num_grad_updates=self.num_grad_updates)

    @Deprecated(error=False)
    def _get_slice_indices(self, slice_size):
        data_slices = []
        data_slices_states = []
        if self.get(SampleBatch.SEQ_LENS) is not None and len(self[SampleBatch.SEQ_LENS]) > 0:
            assert np.all(self[SampleBatch.SEQ_LENS] < slice_size), 'ERROR: `slice_size` must be larger than the max. seq-len in the batch!'
            start_pos = 0
            current_slize_size = 0
            actual_slice_idx = 0
            start_idx = 0
            idx = 0
            while idx < len(self[SampleBatch.SEQ_LENS]):
                seq_len = self[SampleBatch.SEQ_LENS][idx]
                current_slize_size += seq_len
                actual_slice_idx += seq_len if not self.zero_padded else self.max_seq_len
                if current_slize_size >= slice_size:
                    end_idx = idx + 1
                    if not self.zero_padded:
                        data_slices.append((start_pos, start_pos + slice_size))
                        start_pos += slice_size
                        if current_slize_size > slice_size:
                            overhead = current_slize_size - slice_size
                            start_pos -= seq_len - overhead
                            idx -= 1
                    else:
                        data_slices.append((start_pos, actual_slice_idx))
                        start_pos = actual_slice_idx
                    data_slices_states.append((start_idx, end_idx))
                    current_slize_size = 0
                    start_idx = idx + 1
                idx += 1
        else:
            i = 0
            while i < self.count:
                data_slices.append((i, i + slice_size))
                i += slice_size
        return (data_slices, data_slices_states)

    @ExperimentalAPI
    def get_single_step_input_dict(self, view_requirements: ViewRequirementsDict, index: Union[str, int]='last') -> 'SampleBatch':
        """Creates single ts SampleBatch at given index from `self`.

        For usage as input-dict for model (action or value function) calls.

        Args:
            view_requirements: A view requirements dict from the model for
                which to produce the input_dict.
            index: An integer index value indicating the
                position in the trajectory for which to generate the
                compute_actions input dict. Set to "last" to generate the dict
                at the very end of the trajectory (e.g. for value estimation).
                Note that "last" is different from -1, as "last" will use the
                final NEXT_OBS as observation input.

        Returns:
            The (single-timestep) input dict for ModelV2 calls.
        """
        last_mappings = {SampleBatch.OBS: SampleBatch.NEXT_OBS, SampleBatch.PREV_ACTIONS: SampleBatch.ACTIONS, SampleBatch.PREV_REWARDS: SampleBatch.REWARDS}
        input_dict = {}
        for view_col, view_req in view_requirements.items():
            if view_req.used_for_compute_actions is False:
                continue
            data_col = view_req.data_col or view_col
            if index == 'last':
                data_col = last_mappings.get(data_col, data_col)
                if view_req.shift_from is not None:
                    data = self[view_col][-1]
                    traj_len = len(self[data_col])
                    missing_at_end = traj_len % view_req.batch_repeat_value
                    obs_shift = -1 if data_col in [SampleBatch.OBS, SampleBatch.NEXT_OBS] else 0
                    from_ = view_req.shift_from + obs_shift
                    to_ = view_req.shift_to + obs_shift + 1
                    if to_ == 0:
                        to_ = None
                    input_dict[view_col] = np.array([np.concatenate([data, self[data_col][-missing_at_end:]])[from_:to_]])
                else:
                    input_dict[view_col] = tree.map_structure(lambda v: v[-1:], self[data_col])
            else:
                input_dict[view_col] = self[data_col][index:index + 1 if index != -1 else None]
        return SampleBatch(input_dict, seq_lens=np.array([1], dtype=np.int32))