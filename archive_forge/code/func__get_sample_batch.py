import copy
import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def _get_sample_batch(self, batch_data: Dict[str, TensorType]) -> SampleBatch:
    """Returns a SampleBatch from the given data dictionary. Also updates the
        sequence information based on the max_seq_len."""
    batch = SampleBatch(batch_data, is_training=self.training)
    if self.is_policy_recurrent:
        seq_lens = []
        max_seq_len = self.max_seq_len
        count = batch.count
        while count > 0:
            seq_lens.append(min(count, max_seq_len))
            count -= max_seq_len
        batch['seq_lens'] = np.array(seq_lens)
        batch.max_seq_len = max_seq_len
    return batch