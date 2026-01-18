from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@kullback_leibler.RegisterKL(Categorical, Categorical)
def _kl_categorical_categorical(a, b, name=None):
    """Calculate the batched KL divergence KL(a || b) with a and b Categorical.

  Args:
    a: instance of a Categorical distribution object.
    b: instance of a Categorical distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_categorical_categorical".

  Returns:
    Batchwise KL(a || b)
  """
    with ops.name_scope(name, 'kl_categorical_categorical', values=[a.logits, b.logits]):
        delta_log_probs1 = nn_ops.log_softmax(a.logits) - nn_ops.log_softmax(b.logits)
        return math_ops.reduce_sum(nn_ops.softmax(a.logits) * delta_log_probs1, axis=-1)