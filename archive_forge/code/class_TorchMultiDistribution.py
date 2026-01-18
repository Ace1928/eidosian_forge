import gymnasium as gym
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import tree
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, Union, Tuple
@DeveloperAPI
class TorchMultiDistribution(Distribution):
    """Action distribution that operates on multiple, possibly nested actions."""

    def __init__(self, child_distribution_struct: Union[Tuple, List, Dict]):
        """Initializes a TorchMultiActionDistribution object.

        Args:
            child_distribution_struct: Any struct
                that contains the child distribution classes to use to
                instantiate the child distributions from `logits`.
        """
        super().__init__()
        self._original_struct = child_distribution_struct
        self._flat_child_distributions = tree.flatten(child_distribution_struct)

    @override(Distribution)
    def rsample(self, *, sample_shape: Tuple[int, ...]=None, **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        rsamples = []
        for dist in self._flat_child_distributions:
            rsample = dist.rsample(sample_shape=sample_shape, **kwargs)
            rsamples.append(rsample)
        rsamples = tree.unflatten_as(self._original_struct, rsamples)
        return rsamples

    @override(Distribution)
    def logp(self, value: TensorType) -> TensorType:
        if isinstance(value, torch.Tensor):
            split_indices = []
            for dist in self._flat_child_distributions:
                if isinstance(dist, TorchCategorical):
                    split_indices.append(1)
                elif isinstance(dist, TorchMultiCategorical):
                    split_indices.append(len(dist._cats))
                else:
                    sample = dist.sample()
                    if len(sample.shape) == 1:
                        split_indices.append(1)
                    else:
                        split_indices.append(sample.size()[1])
            split_value = list(torch.split(value, split_indices, dim=1))
        else:
            split_value = tree.flatten(value)

        def map_(val, dist):
            if isinstance(dist, TorchCategorical) and val.shape[-1] == 1 and (len(val.shape) > 1):
                val = torch.squeeze(val, dim=-1)
            return dist.logp(val)
        flat_logps = tree.map_structure(map_, split_value, self._flat_child_distributions)
        return sum(flat_logps)

    @override(Distribution)
    def kl(self, other: Distribution) -> TensorType:
        kl_list = [d.kl(o) for d, o in zip(self._flat_child_distributions, other._flat_child_distributions)]
        return sum(kl_list)

    @override(Distribution)
    def entropy(self):
        entropy_list = [d.entropy() for d in self._flat_child_distributions]
        return sum(entropy_list)

    @override(Distribution)
    def sample(self):
        child_distributions_struct = tree.unflatten_as(self._original_struct, self._flat_child_distributions)
        return tree.map_structure(lambda s: s.sample(), child_distributions_struct)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, input_lens: List[int], **kwargs) -> int:
        return sum(input_lens)

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: torch.Tensor, child_distribution_cls_struct: Union[Mapping, Iterable], input_lens: Union[Dict, List[int]], space: gym.Space, **kwargs) -> 'TorchMultiDistribution':
        """Creates this Distribution from logits (and additional arguments).

        If you wish to create this distribution from logits only, please refer to
        `Distribution.get_partial_dist_cls()`.

        Args:
            logits: The tensor containing logits to be separated by `input_lens`.
                child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            input_lens: A list or dict of integers that indicate the length of each
                logit. If this is given as a dict, the structure should match the
                structure of child_distribution_cls_struct.
            space: The possibly nested output space.
            **kwargs: Forward compatibility kwargs.

        Returns:
            A TorchMultiActionDistribution object.
        """
        logit_lens = tree.flatten(input_lens)
        child_distribution_cls_list = tree.flatten(child_distribution_cls_struct)
        split_logits = torch.split(logits, logit_lens, dim=1)
        child_distribution_list = tree.map_structure(lambda dist, input_: dist.from_logits(input_), child_distribution_cls_list, list(split_logits))
        child_distribution_struct = tree.unflatten_as(child_distribution_cls_struct, child_distribution_list)
        return TorchMultiDistribution(child_distribution_struct=child_distribution_struct)

    def to_deterministic(self) -> 'TorchMultiDistribution':
        flat_deterministic_dists = [dist.to_deterministic() for dist in self._flat_child_distributions]
        deterministic_dists = tree.unflatten_as(self._original_struct, flat_deterministic_dists)
        return TorchMultiDistribution(deterministic_dists)