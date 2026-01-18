import collections
import logging
import numpy as np
from typing import List, Any, Dict, Optional, TYPE_CHECKING
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.util.debug import log_once
@Deprecated(new='a child class of `SampleCollector`', error=False)
class SampleBatchBuilder:
    """Util to build a SampleBatch incrementally.

    For efficiency, SampleBatches hold values in column form (as arrays).
    However, it is useful to add data one row (dict) at a time.
    """
    _next_unroll_id = 0

    def __init__(self):
        self.buffers: Dict[str, List] = collections.defaultdict(list)
        self.count = 0

    def add_values(self, **values: Any) -> None:
        """Add the given dictionary (row) of values to this batch."""
        for k, v in values.items():
            self.buffers[k].append(v)
        self.count += 1

    def add_batch(self, batch: SampleBatch) -> None:
        """Add the given batch of values to this batch."""
        for k, column in batch.items():
            self.buffers[k].extend(column)
        self.count += batch.count

    def build_and_reset(self) -> SampleBatch:
        """Returns a sample batch including all previously added values."""
        batch = SampleBatch({k: _to_float_array(v) for k, v in self.buffers.items()})
        if SampleBatch.UNROLL_ID not in batch:
            batch[SampleBatch.UNROLL_ID] = np.repeat(SampleBatchBuilder._next_unroll_id, batch.count)
            SampleBatchBuilder._next_unroll_id += 1
        self.buffers.clear()
        self.count = 0
        return batch