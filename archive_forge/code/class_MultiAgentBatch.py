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
class MultiAgentBatch:
    """A batch of experiences from multiple agents in the environment.

    Attributes:
        policy_batches (Dict[PolicyID, SampleBatch]): Mapping from policy
            ids to SampleBatches of experiences.
        count: The number of env steps in this batch.
    """

    @PublicAPI
    def __init__(self, policy_batches: Dict[PolicyID, SampleBatch], env_steps: int):
        """Initialize a MultiAgentBatch instance.

        Args:
            policy_batches: Mapping from policy
                ids to SampleBatches of experiences.
            env_steps: The number of environment steps in the environment
                this batch contains. This will be less than the number of
                transitions this batch contains across all policies in total.
        """
        for v in policy_batches.values():
            assert isinstance(v, SampleBatch)
        self.policy_batches = policy_batches
        self.count = env_steps

    @PublicAPI
    def env_steps(self) -> int:
        """The number of env steps (there are >= 1 agent steps per env step).

        Returns:
            The number of environment steps contained in this batch.
        """
        return self.count

    @PublicAPI
    def __len__(self) -> int:
        """Same as `self.env_steps()`."""
        return self.count

    @PublicAPI
    def agent_steps(self) -> int:
        """The number of agent steps (there are >= 1 agent steps per env step).

        Returns:
            The number of agent steps total in this batch.
        """
        ct = 0
        for batch in self.policy_batches.values():
            ct += batch.count
        return ct

    @PublicAPI
    def timeslices(self, k: int) -> List['MultiAgentBatch']:
        """Returns k-step batches holding data for each agent at those steps.

        For examples, suppose we have agent1 observations [a1t1, a1t2, a1t3],
        for agent2, [a2t1, a2t3], and for agent3, [a3t3] only.

        Calling timeslices(1) would return three MultiAgentBatches containing
        [a1t1, a2t1], [a1t2], and [a1t3, a2t3, a3t3].

        Calling timeslices(2) would return two MultiAgentBatches containing
        [a1t1, a1t2, a2t1], and [a1t3, a2t3, a3t3].

        This method is used to implement "lockstep" replay mode. Note that this
        method does not guarantee each batch contains only data from a single
        unroll. Batches might contain data from multiple different envs.
        """
        from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
        steps = []
        for policy_id, batch in self.policy_batches.items():
            for row in batch.rows():
                steps.append((row[SampleBatch.EPS_ID], row[SampleBatch.T], row[SampleBatch.AGENT_INDEX], policy_id, row))
        steps.sort()
        finished_slices = []
        cur_slice = collections.defaultdict(SampleBatchBuilder)
        cur_slice_size = 0

        def finish_slice():
            nonlocal cur_slice_size
            assert cur_slice_size > 0
            batch = MultiAgentBatch({k: v.build_and_reset() for k, v in cur_slice.items()}, cur_slice_size)
            cur_slice_size = 0
            cur_slice.clear()
            finished_slices.append(batch)
        for _, group in itertools.groupby(steps, lambda x: x[:2]):
            for _, _, _, policy_id, row in group:
                cur_slice[policy_id].add_values(**row)
            cur_slice_size += 1
            if cur_slice_size >= k:
                finish_slice()
                assert cur_slice_size == 0
        if cur_slice_size > 0:
            finish_slice()
        assert len(finished_slices) > 0, finished_slices
        return finished_slices

    @staticmethod
    @PublicAPI
    def wrap_as_needed(policy_batches: Dict[PolicyID, SampleBatch], env_steps: int) -> Union[SampleBatch, 'MultiAgentBatch']:
        """Returns SampleBatch or MultiAgentBatch, depending on given policies.
        If policy_batches is empty (i.e. {}) it returns an empty MultiAgentBatch.

        Args:
            policy_batches: Mapping from policy ids to SampleBatch.
            env_steps: Number of env steps in the batch.

        Returns:
            The single default policy's SampleBatch or a MultiAgentBatch
            (more than one policy).
        """
        if len(policy_batches) == 1 and DEFAULT_POLICY_ID in policy_batches:
            return policy_batches[DEFAULT_POLICY_ID]
        return MultiAgentBatch(policy_batches=policy_batches, env_steps=env_steps)

    @staticmethod
    @PublicAPI
    @Deprecated(new='concat_samples() from rllib.policy.sample_batch', error=True)
    def concat_samples(samples: List['MultiAgentBatch']) -> 'MultiAgentBatch':
        return concat_samples_into_ma_batch(samples)

    @PublicAPI
    def copy(self) -> 'MultiAgentBatch':
        """Deep-copies self into a new MultiAgentBatch.

        Returns:
            The copy of self with deep-copied data.
        """
        return MultiAgentBatch({k: v.copy() for k, v in self.policy_batches.items()}, self.count)

    @PublicAPI
    def size_bytes(self) -> int:
        """
        Returns:
            The overall size in bytes of all policy batches (all columns).
        """
        return sum((b.size_bytes() for b in self.policy_batches.values()))

    @DeveloperAPI
    def compress(self, bulk: bool=False, columns: Set[str]=frozenset(['obs', 'new_obs'])) -> None:
        """Compresses each policy batch (per column) in place.

        Args:
            bulk: Whether to compress across the batch dimension (0)
                as well. If False will compress n separate list items, where n
                is the batch size.
            columns: Set of column names to compress.
        """
        for batch in self.policy_batches.values():
            batch.compress(bulk=bulk, columns=columns)

    @DeveloperAPI
    def decompress_if_needed(self, columns: Set[str]=frozenset(['obs', 'new_obs'])) -> 'MultiAgentBatch':
        """Decompresses each policy batch (per column), if already compressed.

        Args:
            columns: Set of column names to decompress.

        Returns:
            Self.
        """
        for batch in self.policy_batches.values():
            batch.decompress_if_needed(columns)
        return self

    @DeveloperAPI
    def as_multi_agent(self) -> 'MultiAgentBatch':
        """Simply returns `self` (already a MultiAgentBatch).

        Returns:
            This very instance of MultiAgentBatch.
        """
        return self

    def __getitem__(self, key: str) -> SampleBatch:
        """Returns the SampleBatch for the given policy id."""
        return self.policy_batches[key]

    def __str__(self):
        return 'MultiAgentBatch({}, env_steps={})'.format(str(self.policy_batches), self.count)

    def __repr__(self):
        return 'MultiAgentBatch({}, env_steps={})'.format(str(self.policy_batches), self.count)