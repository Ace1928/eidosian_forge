import logging
import math
from pathlib import Path
import re
import numpy as np
from typing import List, Tuple, TYPE_CHECKING, Optional
import zipfile
import ray.data
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_reader import from_json_data, postprocess_actions
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType
def _postprocess_if_needed(self, batch: SampleBatchType) -> SampleBatchType:
    if not self._ioctx.config.get('postprocess_inputs'):
        return batch
    if isinstance(batch, SampleBatch):
        out = []
        for sub_batch in batch.split_by_episode():
            if self._default_policy is not None:
                out.append(self._default_policy.postprocess_trajectory(sub_batch))
            else:
                out.append(sub_batch)
        return concat_samples(out)
    else:
        raise NotImplementedError('Postprocessing of multi-agent data not implemented yet.')