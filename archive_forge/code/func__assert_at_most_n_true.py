import collections
import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _assert_at_most_n_true(predicates, n, msg):
    """Returns an Assert op that checks that at most n predicates are True.

  Args:
    predicates: list of bool scalar tensors.
    n: maximum number of true predicates allowed.
    msg: Error message.
  """
    preds_c = array_ops_stack.stack(predicates, name='preds_c')
    num_true_conditions = math_ops.reduce_sum(math_ops.cast(preds_c, dtypes.int32), name='num_true_conds')
    condition = math_ops.less_equal(num_true_conditions, constant_op.constant(n, name='n_true_conds'))
    preds_names = ', '.join((getattr(p, 'name', '?') for p in predicates))
    error_msg = ['%s: more than %d conditions (%s) evaluated as True:' % (msg, n, preds_names), preds_c]
    return control_flow_assert.Assert(condition, data=error_msg, summarize=len(predicates))