import gymnasium as gym
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Type, Union
import ray
from ray.rllib.algorithms.appo.utils import make_appo_models
import ray.rllib.algorithms.impala.vtrace_torch as vtrace
from ray.rllib.algorithms.impala.impala_torch_policy import (
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import (
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import TensorType
Stats function for APPO. Returns a dict with important loss stats.

        Args:
            policy: The Policy to generate stats for.
            train_batch: The SampleBatch (already) used for training.

        Returns:
            Dict[str, TensorType]: The stats dict.
        