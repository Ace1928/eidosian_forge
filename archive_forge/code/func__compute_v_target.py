import logging
from typing import Dict, Any, Optional, List
import math
import numpy as np
from ray.data import Dataset
from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimator
from ray.rllib.offline.offline_evaluation_utils import compute_q_and_v_values
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.numpy import convert_to_numpy
def _compute_v_target(self, init_step):
    v_target = self.model.estimate_v(init_step)
    v_target = convert_to_numpy(v_target)
    return v_target