import copy
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any
import ray
from ray.data import Dataset
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, convert_ma_batch_to_sample_batch
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
@DeveloperAPI
def _perturb_fn(batch: np.ndarray, index: int):
    random_inds = np.random.permutation(batch.shape[0])
    batch[:, index] = batch[random_inds, index]