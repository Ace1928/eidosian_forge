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
class TorchMultiCategorical(TorchDistributionWrapper):
    """MultiCategorical distribution for MultiDiscrete action spaces."""

    @override(TorchDistributionWrapper)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2, input_lens: Union[List[int], np.ndarray, Tuple[int, ...]], action_space=None):
        super().__init__(inputs, model)
        inputs_split = self.inputs.split(tuple(input_lens), dim=1)
        self.cats = [torch.distributions.categorical.Categorical(logits=input_) for input_ in inputs_split]
        self.action_space = action_space

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        arr = [cat.sample() for cat in self.cats]
        sample_ = torch.stack(arr, dim=1)
        if isinstance(self.action_space, gym.spaces.Box):
            sample_ = torch.reshape(sample_, [-1] + list(self.action_space.shape))
        self.last_sample = sample_
        return sample_

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        arr = [torch.argmax(cat.probs, -1) for cat in self.cats]
        sample_ = torch.stack(arr, dim=1)
        if isinstance(self.action_space, gym.spaces.Box):
            sample_ = torch.reshape(sample_, [-1] + list(self.action_space.shape))
        self.last_sample = sample_
        return sample_

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        if isinstance(actions, torch.Tensor):
            if isinstance(self.action_space, gym.spaces.Box):
                actions = torch.reshape(actions, [-1, int(np.prod(self.action_space.shape))])
            actions = torch.unbind(actions, dim=1)
        logps = torch.stack([cat.log_prob(act) for cat, act in zip(self.cats, actions)])
        return torch.sum(logps, dim=0)

    @override(ActionDistribution)
    def multi_entropy(self) -> TensorType:
        return torch.stack([cat.entropy() for cat in self.cats], dim=1)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return torch.sum(self.multi_entropy(), dim=1)

    @override(ActionDistribution)
    def multi_kl(self, other: ActionDistribution) -> TensorType:
        return torch.stack([torch.distributions.kl.kl_divergence(cat, oth_cat) for cat, oth_cat in zip(self.cats, other.cats)], dim=1)

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        return torch.sum(self.multi_kl(other), dim=1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        if isinstance(action_space, gym.spaces.Box):
            assert action_space.dtype.name.startswith('int')
            low_ = np.min(action_space.low)
            high_ = np.max(action_space.high)
            assert np.all(action_space.low == low_)
            assert np.all(action_space.high == high_)
            return np.prod(action_space.shape, dtype=np.int32) * (high_ - low_ + 1)
        else:
            return np.sum(action_space.nvec)