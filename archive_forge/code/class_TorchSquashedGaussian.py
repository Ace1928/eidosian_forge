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
class TorchSquashedGaussian(TorchDistributionWrapper):
    """A tanh-squashed Gaussian distribution defined by: mean, std, low, high.

    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """

    def __init__(self, inputs: List[TensorType], model: TorchModelV2, low: float=-1.0, high: float=1.0):
        """Parameterizes the distribution via `inputs`.

        Args:
            low: The lowest possible sampling value
                (excluding this value).
            high: The highest possible sampling value
                (excluding this value).
        """
        super().__init__(inputs, model)
        mean, log_std = torch.chunk(self.inputs, 2, dim=-1)
        log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        std = torch.exp(log_std)
        self.dist = torch.distributions.normal.Normal(mean, std)
        assert np.all(np.less(low, high))
        self.low = low
        self.high = high
        self.mean = mean
        self.std = std

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        normal_sample = self.dist.rsample()
        self.last_sample = self._squash(normal_sample)
        return self.last_sample

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        unsquashed_values = self._unsquash(x)
        log_prob_gaussian = self.dist.log_prob(unsquashed_values)
        log_prob_gaussian = torch.clamp(log_prob_gaussian, -100, 100)
        log_prob_gaussian = torch.sum(log_prob_gaussian, dim=-1)
        unsquashed_values_tanhd = torch.tanh(unsquashed_values)
        log_prob = log_prob_gaussian - torch.sum(torch.log(1 - unsquashed_values_tanhd ** 2 + SMALL_NUMBER), dim=-1)
        return log_prob

    def sample_logp(self):
        z = self.dist.rsample()
        actions = self._squash(z)
        return (actions, torch.sum(self.dist.log_prob(z) - torch.log(1 - actions * actions + SMALL_NUMBER), dim=-1))

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        raise ValueError('Entropy not defined for SquashedGaussian!')

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        raise ValueError('KL not defined for SquashedGaussian!')

    def _squash(self, raw_values: TensorType) -> TensorType:
        squashed = (torch.tanh(raw_values) + 1.0) / 2.0 * (self.high - self.low) + self.low
        return torch.clamp(squashed, self.low, self.high)

    def _unsquash(self, values: TensorType) -> TensorType:
        normed_values = (values - self.low) / (self.high - self.low) * 2.0 - 1.0
        save_normed_values = torch.clamp(normed_values, -1.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER)
        unsquashed = torch.atanh(save_normed_values)
        return unsquashed

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape, dtype=np.int32) * 2