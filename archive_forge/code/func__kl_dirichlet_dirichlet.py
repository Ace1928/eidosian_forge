import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@kullback_leibler.RegisterKL(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(d1, d2, name=None):
    """Batchwise KL divergence KL(d1 || d2) with d1 and d2 Dirichlet.

  Args:
    d1: instance of a Dirichlet distribution object.
    d2: instance of a Dirichlet distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_dirichlet_dirichlet".

  Returns:
    Batchwise KL(d1 || d2)
  """
    with ops.name_scope(name, 'kl_dirichlet_dirichlet', values=[d1.concentration, d2.concentration]):
        digamma_sum_d1 = math_ops.digamma(math_ops.reduce_sum(d1.concentration, axis=-1, keepdims=True))
        digamma_diff = math_ops.digamma(d1.concentration) - digamma_sum_d1
        concentration_diff = d1.concentration - d2.concentration
        return math_ops.reduce_sum(concentration_diff * digamma_diff, axis=-1) - special_math_ops.lbeta(d1.concentration) + special_math_ops.lbeta(d2.concentration)