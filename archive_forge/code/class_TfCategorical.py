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
class TfCategorical(TfDistribution):
    """Wrapper class for Categorical distribution.

    Creates a categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    Samples are integers from :math:`\\{0, \\ldots, K-1\\}` where `K` is
    ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative
    probability of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. testcode::
        :skipif: True

        m = TfCategorical([ 0.25, 0.25, 0.25, 0.25 ])
        m.sample(sample_shape=(2,))  # equal probability of 0, 1, 2, 3

    .. testoutput::

        tf.Tensor([2 3], shape=(2,), dtype=int32)

    Args:
        probs: The probablities of each event.
        logits: Event log probabilities (unnormalized)
        temperature: In case of using logits, this parameter can be used to determine
            the sharpness of the distribution. i.e.
            ``probs = softmax(logits / temperature)``. The temperature must be strictly
            positive. A low value (e.g. 1e-10) will result in argmax sampling while a
            larger value will result in uniform sampling.
    """

    @override(TfDistribution)
    def __init__(self, probs: 'tf.Tensor'=None, logits: 'tf.Tensor'=None) -> None:
        assert (probs is None) != (logits is None), 'Exactly one out of `probs` and `logits` must be set!'
        self.probs = probs
        self.logits = logits
        self.one_hot = tfp.distributions.OneHotCategorical(logits=logits, probs=probs)
        super().__init__(logits=logits, probs=probs)

    @override(Distribution)
    def logp(self, value: TensorType, **kwargs) -> TensorType:
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits if self.logits is not None else tf.log(self.probs), labels=tf.cast(value, tf.int32))

    @override(TfDistribution)
    def _get_tf_distribution(self, probs: 'tf.Tensor'=None, logits: 'tf.Tensor'=None) -> 'tfp.distributions.Distribution':
        return tfp.distributions.Categorical(probs=probs, logits=logits)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        assert isinstance(space, gym.spaces.Discrete)
        return int(space.n)

    @override(Distribution)
    def rsample(self, sample_shape=()):
        one_hot_sample = self.one_hot.sample(sample_shape)
        return tf.stop_gradients(one_hot_sample - self.probs) + self.probs

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'TfCategorical':
        return TfCategorical(logits=logits, **kwargs)

    def to_deterministic(self) -> 'TfDeterministic':
        if self.probs is not None:
            probs_or_logits = self.probs
        else:
            probs_or_logits = self.logits
        return TfDeterministic(loc=tf.math.argmax(probs_or_logits, axis=-1))