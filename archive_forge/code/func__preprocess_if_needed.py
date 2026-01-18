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
def _preprocess_if_needed(self, batch: SampleBatchType) -> SampleBatchType:
    if self.preprocessor:
        for key in (SampleBatch.CUR_OBS, SampleBatch.NEXT_OBS):
            if key in batch:
                batch[key] = np.stack([self.preprocessor.transform(s) for s in batch[key]])
    return batch