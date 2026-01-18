from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
def _maybe_expand_weights():
    expand_weights = lambda: array_ops.expand_dims(sample_weight, [-1])
    return cond.cond(math_ops.equal(rank_diff, -1), expand_weights, lambda: sample_weight)