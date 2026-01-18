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
class TfMultiCategorical(Distribution):
    """MultiCategorical distribution for MultiDiscrete action spaces."""

    @override(Distribution)
    def __init__(self, categoricals: List[TfCategorical]):
        super().__init__()
        self._cats = categoricals

    @override(Distribution)
    def sample(self) -> TensorType:
        arr = [cat.sample() for cat in self._cats]
        sample_ = tf.stack(arr, axis=-1)
        return sample_

    @override(Distribution)
    def rsample(self, sample_shape=()):
        arr = [cat.rsample() for cat in self._cats]
        sample_ = tf.stack(arr, axis=-1)
        return sample_

    @override(Distribution)
    def logp(self, value: tf.Tensor) -> TensorType:
        actions = tf.unstack(tf.cast(value, tf.int32), axis=-1)
        logps = tf.stack([cat.logp(act) for cat, act in zip(self._cats, actions)])
        return tf.reduce_sum(logps, axis=0)

    @override(Distribution)
    def entropy(self) -> TensorType:
        return tf.reduce_sum(tf.stack([cat.entropy() for cat in self._cats], axis=-1), axis=-1)

    @override(Distribution)
    def kl(self, other: Distribution) -> TensorType:
        kls = tf.stack([cat.kl(oth_cat) for cat, oth_cat in zip(self._cats, other._cats)], axis=-1)
        return tf.reduce_sum(kls, axis=-1)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        assert isinstance(space, gym.spaces.MultiDiscrete)
        return int(np.sum(space.nvec))

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: tf.Tensor, input_lens: List[int], **kwargs) -> 'TfMultiCategorical':
        """Creates this Distribution from logits (and additional arguments).

        If you wish to create this distribution from logits only, please refer to
        `Distribution.get_partial_dist_cls()`.

        Args:
            logits: The tensor containing logits to be separated by logit_lens.
                child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            input_lens: A list of integers that indicate the length of the logits
                vectors to be passed into each child distribution.
            **kwargs: Forward compatibility kwargs.
        """
        categoricals = [TfCategorical(logits=logits) for logits in tf.split(logits, input_lens, axis=-1)]
        return TfMultiCategorical(categoricals=categoricals)

    def to_deterministic(self) -> 'TfMultiDistribution':
        return TfMultiDistribution([cat.to_deterministic() for cat in self._cats])