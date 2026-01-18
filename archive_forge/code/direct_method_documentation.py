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
Calculates the Direct Method estimate on the given dataset.

        Note: This estimate works for only discrete action spaces for now.

        Args:
            dataset: Dataset to compute the estimate on. Each record in dataset should
                include the following columns: `obs`, `actions`, `action_prob` and
                `rewards`. The `obs` on each row shoud be a vector of D dimensions.
            n_parallelism: The number of parallel workers to use.

        Returns:
            Dictionary with the following keys:
                v_target: The estimated value of the target policy.
                v_behavior: The estimated value of the behavior policy.
                v_gain: The estimated gain of the target policy over the behavior
                    policy.
                v_std: The standard deviation of the estimated value of the target.
        