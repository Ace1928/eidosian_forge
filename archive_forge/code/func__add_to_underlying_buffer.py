from typing import Dict
import logging
import numpy as np
from ray.util.timer import _Timer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import (
from ray.rllib.utils.replay_buffers.replay_buffer import (
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.util.debug import log_once
from ray.util.annotations import DeveloperAPI
from ray.rllib.policy.rnn_sequencing import timeslice_along_seq_lens_with_overlap
@DeveloperAPI
@override(MultiAgentReplayBuffer)
def _add_to_underlying_buffer(self, policy_id: PolicyID, batch: SampleBatchType, **kwargs) -> None:
    """Add a batch of experiences to the underlying buffer of a policy.

        If the storage unit is `timesteps`, cut the batch into timeslices
        before adding them to the appropriate buffer. Otherwise, let the
        underlying buffer decide how slice batches.

        Args:
            policy_id: ID of the policy that corresponds to the underlying
            buffer
            batch: SampleBatch to add to the underlying buffer
            ``**kwargs``: Forward compatibility kwargs.
        """
    kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)
    if self.storage_unit is StorageUnit.TIMESTEPS:
        timeslices = batch.timeslices(1)
    elif self.storage_unit is StorageUnit.SEQUENCES:
        timeslices = timeslice_along_seq_lens_with_overlap(sample_batch=batch, seq_lens=batch.get(SampleBatch.SEQ_LENS) if self.replay_sequence_override else None, zero_pad_max_seq_len=self.replay_sequence_length, pre_overlap=self.replay_burn_in, zero_init_states=self.replay_zero_init_states)
    elif self.storage_unit == StorageUnit.EPISODES:
        timeslices = []
        for eps in batch.split_by_episode():
            if eps.get(SampleBatch.T)[0] == 0 and (eps.get(SampleBatch.TERMINATEDS, [True])[-1] or eps.get(SampleBatch.TRUNCATEDS, [False])[-1]):
                timeslices.append(eps)
            elif log_once('only_full_episodes'):
                logger.info('This buffer uses episodes as a storage unit and thus allows only full episodes to be added to it. Some samples may be dropped.')
    elif self.storage_unit == StorageUnit.FRAGMENTS:
        timeslices = [batch]
    else:
        raise ValueError('Unknown `storage_unit={}`'.format(self.storage_unit))
    for slice in timeslices:
        if self.replay_mode is ReplayMode.INDEPENDENT:
            if 'weights' in slice and len(slice['weights']):
                weight = np.mean(slice['weights'])
            else:
                weight = None
            if 'weight' in kwargs and weight is not None:
                if log_once('overwrite_weight'):
                    logger.warning('Adding batches with column `weights` to this buffer while providing weights as a call argument to the add method results in the column being overwritten.')
            kwargs = {'weight': weight, **kwargs}
        else:
            if 'weight' in kwargs:
                if log_once('lockstep_no_weight_allowed'):
                    logger.warning('Settings weights for batches in lockstep mode is not allowed.Weights are being ignored.')
            kwargs = {**kwargs, 'weight': None}
        self.replay_buffers[policy_id].add(slice, **kwargs)