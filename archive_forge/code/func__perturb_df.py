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
@ExperimentalAPI
def _perturb_df(batch: pd.DataFrame, index: int):
    obs_batch = np.vstack(batch['obs'].values)
    _perturb_fn(obs_batch, index)
    batch['perturbed_obs'] = list(obs_batch)
    return batch