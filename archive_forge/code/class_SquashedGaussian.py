import functools
import gymnasium as gym
from math import log
import numpy as np
import tree  # pip install dm_tree
from typing import Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, SMALL_NUMBER
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
@DeveloperAPI
class SquashedGaussian(TFActionDistribution):
    """A tanh-squashed Gaussian distribution defined by: mean, std, low, high.

    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """

    def __init__(self, inputs: List[TensorType], model: ModelV2, low: float=-1.0, high: float=1.0):
        """Parameterizes the distribution via `inputs`.

        Args:
            low: The lowest possible sampling value
                (excluding this value).
            high: The highest possible sampling value
                (excluding this value).
        """
        assert tfp is not None
        mean, log_std = tf.split(inputs, 2, axis=-1)
        log_std = tf.clip_by_value(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        std = tf.exp(log_std)
        self.distr = tfp.distributions.Normal(loc=mean, scale=std)
        assert np.all(np.less(low, high))
        self.low = low
        self.high = high
        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        mean = self.distr.mean()
        return self._squash(mean)

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        return self._squash(self.distr.sample())

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        unsquashed_values = tf.cast(self._unsquash(x), self.inputs.dtype)
        log_prob_gaussian = self.distr.log_prob(unsquashed_values)
        log_prob_gaussian = tf.clip_by_value(log_prob_gaussian, -100, 100)
        log_prob_gaussian = tf.reduce_sum(log_prob_gaussian, axis=-1)
        unsquashed_values_tanhd = tf.math.tanh(unsquashed_values)
        log_prob = log_prob_gaussian - tf.reduce_sum(tf.math.log(1 - unsquashed_values_tanhd ** 2 + SMALL_NUMBER), axis=-1)
        return log_prob

    def sample_logp(self):
        z = self.distr.sample()
        actions = self._squash(z)
        return (actions, tf.reduce_sum(self.distr.log_prob(z) - tf.math.log(1 - actions * actions + SMALL_NUMBER), axis=-1))

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        raise ValueError('Entropy not defined for SquashedGaussian!')

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        raise ValueError('KL not defined for SquashedGaussian!')

    def _squash(self, raw_values: TensorType) -> TensorType:
        squashed = (tf.math.tanh(raw_values) + 1.0) / 2.0 * (self.high - self.low) + self.low
        return tf.clip_by_value(squashed, self.low, self.high)

    def _unsquash(self, values: TensorType) -> TensorType:
        normed_values = (values - self.low) / (self.high - self.low) * 2.0 - 1.0
        save_normed_values = tf.clip_by_value(normed_values, -1.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER)
        unsquashed = tf.math.atanh(save_normed_values)
        return unsquashed

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape, dtype=np.int32) * 2