import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
class ClippedIdentity(tfb.identity.Identity):
    """Compute Y = clip_by_value(X, -1, 1).

  Note that we do not override `is_injective` despite this bijector not being
  injective, to not disable Identity's `forward_log_det_jacobian`. See also
  tensorflow_probability.bijectors.identity.Identity.
  """

    def __init__(self, validate_args=False, name='clipped_identity'):
        with tf.name_scope(name) as name:
            super(ClippedIdentity, self).__init__(validate_args=validate_args, name=name)

    @classmethod
    def _is_increasing(cls):
        return False

    def _forward(self, x):
        return tf.clip_by_value(x, -1.0, 1.0)