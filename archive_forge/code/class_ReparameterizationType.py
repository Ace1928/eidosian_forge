import abc
import contextlib
import types
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['distributions.ReparameterizationType'])
class ReparameterizationType:
    """Instances of this class represent how sampling is reparameterized.

  Two static instances exist in the distributions library, signifying
  one of two possible properties for samples from a distribution:

  `FULLY_REPARAMETERIZED`: Samples from the distribution are fully
    reparameterized, and straight-through gradients are supported.

  `NOT_REPARAMETERIZED`: Samples from the distribution are not fully
    reparameterized, and straight-through gradients are either partially
    unsupported or are not supported at all. In this case, for purposes of
    e.g. RL or variational inference, it is generally safest to wrap the
    sample results in a `stop_gradients` call and use policy
    gradients / surrogate loss instead.
  """

    @deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
    def __init__(self, rep_type):
        self._rep_type = rep_type

    def __repr__(self):
        return '<Reparameterization Type: %s>' % self._rep_type

    def __eq__(self, other):
        """Determine if this `ReparameterizationType` is equal to another.

    Since ReparameterizationType instances are constant static global
    instances, equality checks if two instances' id() values are equal.

    Args:
      other: Object to compare against.

    Returns:
      `self is other`.
    """
        return self is other