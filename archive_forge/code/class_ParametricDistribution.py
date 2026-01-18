import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
class ParametricDistribution(abc.ABC):
    """Abstract class for parametric (action) distribution."""

    def __init__(self, param_size, create_dist):
        """Abstract class for parametric (action) distribution.

    Specifies how to transform distribution parameters (i.e. actor output)
    into a distribution over actions.

    Args:
      param_size: Size of the parameters for the distribution
      create_dist: Function from parameters to tf Distribution.
    """
        self._param_size = param_size
        self._create_dist = create_dist

    @property
    def create_dist(self):
        return self._create_dist

    def __call__(self, params):
        return self.create_dist(params)

    @property
    def param_size(self):
        return self._param_size

    @property
    def reparametrizable(self):
        return self._create_dist(tf.zeros((self._param_size,))).reparameterization_type == tfd.FULLY_REPARAMETERIZED

    def sample(self, parameters):
        return self._create_dist(parameters).sample()

    def log_prob(self, parameters, actions):
        return self._create_dist(parameters).log_prob(actions)

    def entropy(self, parameters):
        """Return the entropy of the given distribution."""
        return self._create_dist(parameters).entropy()

    def kl_divergence(self, parameters_a, parameters_b):
        """Return KL divergence between the two distributions."""
        dist_a = self._create_dist(parameters_a)
        dist_b = self._create_dist(parameters_b)
        return tfd.kl_divergence(dist_a, dist_b)