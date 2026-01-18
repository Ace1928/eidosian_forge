from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['losses.Reduction'])
class Reduction:
    """Types of loss reduction.

  Contains the following values:

  * `NONE`: Un-reduced weighted losses with the same shape as input.
  * `SUM`: Scalar sum of weighted losses.
  * `MEAN`: Scalar `SUM` divided by sum of weights. DEPRECATED.
  * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
  * `SUM_OVER_NONZERO_WEIGHTS`: Scalar `SUM` divided by number of non-zero
     weights. DEPRECATED.
  * `SUM_BY_NONZERO_WEIGHTS`: Same as `SUM_OVER_NONZERO_WEIGHTS`. DEPRECATED.
  """
    NONE = 'none'
    SUM = 'weighted_sum'
    SUM_OVER_BATCH_SIZE = 'weighted_sum_over_batch_size'
    MEAN = 'weighted_mean'
    SUM_BY_NONZERO_WEIGHTS = 'weighted_sum_by_nonzero_weights'
    SUM_OVER_NONZERO_WEIGHTS = SUM_BY_NONZERO_WEIGHTS

    @classmethod
    def all(cls):
        return (cls.NONE, cls.SUM, cls.MEAN, cls.SUM_OVER_BATCH_SIZE, cls.SUM_OVER_NONZERO_WEIGHTS, cls.SUM_BY_NONZERO_WEIGHTS)

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError(f'Invalid Reduction Key {key}. Key should be one of {cls.all()}.')