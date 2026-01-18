import functools
import gymnasium as gym
from math import log
import numpy as np
import tree  # pip install dm_tree
from typing import Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
@DeveloperAPI
class TorchDeterministic(TorchDistributionWrapper):
    """Action distribution that returns the input values directly.

    This is similar to DiagGaussian with standard deviation zero (thus only
    requiring the "mean" values as NN output).
    """

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        return self.inputs

    @override(TorchDistributionWrapper)
    def sampled_action_logp(self) -> TensorType:
        return torch.zeros((self.inputs.size()[0],), dtype=torch.float32)

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        return self.deterministic_sample()

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape, dtype=np.int32)