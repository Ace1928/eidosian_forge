from typing import Dict, List, Tuple
import gymnasium as gym
import ray
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import TensorType, AlgorithmConfigDict
Constructs the loss for DQNTorchPolicy.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        train_batch: The training data.

    Returns:
        TensorType: A single loss tensor.
    