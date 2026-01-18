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
class TorchDistributionWrapper(ActionDistribution):
    """Wrapper class for torch.distributions."""

    @override(ActionDistribution)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
            if isinstance(model, TorchModelV2):
                inputs = inputs.to(next(model.parameters()).device)
        super().__init__(inputs, model)
        self.last_sample = None

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        return self.dist.log_prob(actions)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        return self.dist.entropy()

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)

    @override(ActionDistribution)
    def sample(self) -> TensorType:
        self.last_sample = self.dist.sample()
        return self.last_sample

    @override(ActionDistribution)
    def sampled_action_logp(self) -> TensorType:
        assert self.last_sample is not None
        return self.logp(self.last_sample)