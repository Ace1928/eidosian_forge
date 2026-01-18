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
def _case_create_default_action(predicates, actions):
    """Creates default action for a list of actions and their predicates.

  It uses the input actions to select an arbitrary as default and makes sure
  that corresponding predicates have valid values.

  Args:
    predicates: a list of bool scalar tensors
    actions: a list of callable objects which return tensors.

  Returns:
    a callable
  """
    k = len(predicates) - 1
    predicate, action = (predicates[k], actions[k])
    other_predicates, other_actions = (predicates[:k], actions[:k])

    def default_action():
        others_msg = 'Implementation error: selected default action #%d was called, but some of other predicates are True: ' % k
        default_msg = ('Input error: None of conditions evaluated as True:', array_ops_stack.stack(predicates, name='preds_c'))
        with ops.control_dependencies([_assert_at_most_n_true(other_predicates, n=0, msg=others_msg), control_flow_assert.Assert(predicate, data=default_msg)]):
            return action()
    return (default_action, other_predicates, other_actions)