from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@distribution_util.AppendDocstring('The covariance for each batch member is defined as the following:\n\n      ```none\n      Var(X_j) = n * alpha_j / alpha_0 * (1 - alpha_j / alpha_0) *\n      (n + alpha_0) / (1 + alpha_0)\n      ```\n\n      where `concentration = alpha` and\n      `total_concentration = alpha_0 = sum_j alpha_j`.\n\n      The covariance between elements in a batch is defined as:\n\n      ```none\n      Cov(X_i, X_j) = -n * alpha_i * alpha_j / alpha_0 ** 2 *\n      (n + alpha_0) / (1 + alpha_0)\n      ```\n      ')
def _covariance(self):
    x = self._variance_scale_term() * self._mean()
    return array_ops.matrix_set_diag(-math_ops.matmul(x[..., array_ops.newaxis], x[..., array_ops.newaxis, :]), self._variance())