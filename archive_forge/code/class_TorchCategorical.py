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
class TorchCategorical(TorchDistributionWrapper):
    """Wrapper class for PyTorch Categorical distribution."""

    @override(ActionDistribution)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2=None, temperature: float=1.0):
        if temperature != 1.0:
            assert temperature > 0.0, 'Categorical `temperature` must be > 0.0!'
            inputs /= temperature
        super().__init__(inputs, model)
        self.dist = torch.distributions.categorical.Categorical(logits=self.inputs)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = self.dist.probs.argmax(dim=1)
        return self.last_sample

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return action_space.n