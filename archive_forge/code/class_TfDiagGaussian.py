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
class TfDiagGaussian(TfDistribution):
    """Wrapper class for Normal distribution.

    Creates a normal distribution parameterized by :attr:`loc` and :attr:`scale`. In
    case of multi-dimensional distribution, the variance is assumed to be diagonal.

    .. testcode::
        :skipif: True

        m = TfDiagGaussian(loc=[0.0, 0.0], scale=[1.0, 1.0])
        m.sample(sample_shape=(2,))  # 2d normal dist with loc=0 and scale=1

    .. testoutput::

        tensor([[ 0.1046, -0.6120], [ 0.234, 0.556]])

    .. testcode::
        :skipif: True

        # scale is None
        m = TfDiagGaussian(loc=[0.0, 1.0])
        m.sample(sample_shape=(2,))  # normally distributed with loc=0 and scale=1

    .. testoutput::

        tensor([0.1046, 0.6120])


    Args:
        loc: mean of the distribution (often referred to as mu). If scale is None, the
            second half of the `loc` will be used as the log of scale.
        scale: standard deviation of the distribution (often referred to as sigma).
            Has to be positive.
    """

    @override(TfDistribution)
    def __init__(self, loc: Union[float, TensorType], scale: Optional[Union[float, TensorType]]=None):
        self.loc = loc
        super().__init__(loc=loc, scale=scale)

    @override(TfDistribution)
    def _get_tf_distribution(self, loc, scale) -> 'tfp.distributions.Distribution':
        return tfp.distributions.Normal(loc=loc, scale=scale)

    @override(TfDistribution)
    def logp(self, value: TensorType) -> TensorType:
        return tf.math.reduce_sum(super().logp(value), axis=-1)

    @override(TfDistribution)
    def entropy(self) -> TensorType:
        return tf.math.reduce_sum(super().entropy(), axis=-1)

    @override(TfDistribution)
    def kl(self, other: 'TfDistribution') -> TensorType:
        return tf.math.reduce_sum(super().kl(other), axis=-1)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        assert isinstance(space, gym.spaces.Box)
        return int(np.prod(space.shape, dtype=np.int32) * 2)

    @override(Distribution)
    def rsample(self, sample_shape=()):
        eps = tf.random.normal(sample_shape)
        return self._dist.loc + eps * self._dist.scale

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'TfDiagGaussian':
        loc, log_std = tf.split(logits, num_or_size_splits=2, axis=-1)
        scale = tf.math.exp(log_std)
        return TfDiagGaussian(loc=loc, scale=scale)

    def to_deterministic(self) -> 'TfDeterministic':
        return TfDeterministic(loc=self.loc)