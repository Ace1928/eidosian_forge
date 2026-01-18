import gymnasium as gym
import tree
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.typing import TensorType, Union, Tuple
@DeveloperAPI
class TfDeterministic(Distribution):
    """The distribution that returns the input values directly.

    This is similar to DiagGaussian with standard deviation zero (thus only
    requiring the "mean" values as NN output).

    Note: entropy is always zero, ang logp and kl are not implemented.

    .. testcode::
        :skipif: True

        m = TfDeterministic(loc=tf.constant([0.0, 0.0]))
        m.sample(sample_shape=(2,))

    .. testoutput::

        Tensor([[ 0.0, 0.0], [ 0.0, 0.0]])

    Args:
        loc: the determinsitic value to return
    """

    @override(Distribution)
    def __init__(self, loc: 'tf.Tensor') -> None:
        super().__init__()
        self.loc = loc

    @override(Distribution)
    def sample(self, *, sample_shape: Tuple[int, ...]=(), **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        shape = sample_shape + self.loc.shape
        return tf.ones(shape, dtype=self.loc.dtype) * self.loc

    @override(Distribution)
    def rsample(self, *, sample_shape: Tuple[int, ...]=None, **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        raise NotImplementedError

    @override(Distribution)
    def logp(self, value: TensorType, **kwargs) -> TensorType:
        raise ValueError(f'Cannot return logp for {self.__class__.__name__}.')

    @override(Distribution)
    def entropy(self, **kwargs) -> TensorType:
        raise tf.zeros_like(self.loc)

    @override(Distribution)
    def kl(self, other: 'Distribution', **kwargs) -> TensorType:
        raise ValueError(f'Cannot return kl for {self.__class__.__name__}.')

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        assert isinstance(space, gym.spaces.Box)
        return int(np.prod(space.shape, dtype=np.int32))

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'TfDeterministic':
        return TfDeterministic(loc=logits)

    def to_deterministic(self) -> 'TfDeterministic':
        return self