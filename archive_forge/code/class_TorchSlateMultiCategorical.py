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
@ExperimentalAPI
class TorchSlateMultiCategorical(TorchCategorical):
    """MultiCategorical distribution for MultiDiscrete action spaces.

    The action space must be uniform, meaning all nvec items have the same size, e.g.
    MultiDiscrete([10, 10, 10]), where 10 is the number of candidates to pick from
    and 3 is the slate size (pick 3 out of 10). When picking candidates, no candidate
    must be picked more than once.
    """

    def __init__(self, inputs: List[TensorType], model: TorchModelV2=None, temperature: float=1.0, action_space: Optional[gym.spaces.MultiDiscrete]=None, all_slates=None):
        assert temperature > 0.0, 'Categorical `temperature` must be > 0.0!'
        super().__init__(inputs / temperature, model)
        self.action_space = action_space
        assert isinstance(self.action_space, gym.spaces.MultiDiscrete) and all((n == self.action_space.nvec[0] for n in self.action_space.nvec))
        self.all_slates = all_slates

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        sample = super().deterministic_sample()
        return torch.take_along_dim(self.all_slates, sample.long(), dim=-1)

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        return torch.ones_like(self.inputs[:, 0])