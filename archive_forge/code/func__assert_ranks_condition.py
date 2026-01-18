import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _assert_ranks_condition(x, ranks, static_condition, dynamic_condition, data, summarize):
    """Assert `x` has a rank that satisfies a given condition.

  Args:
    x:  Numeric `Tensor`.
    ranks:  Scalar `Tensor`.
    static_condition:   A python function that takes
      `[actual_rank, given_ranks]` and returns `True` if the condition is
      satisfied, `False` otherwise.
    dynamic_condition:  An `op` that takes [actual_rank, given_ranks]
      and return `True` if the condition is satisfied, `False` otherwise.
    data:  The tensors to print out if the condition is false.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.

  Returns:
    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.

  Raises:
    ValueError:  If static checks determine `x` fails static_condition.
  """
    for rank in ranks:
        assert_type(rank, dtypes.int32)
    ranks_static = tuple([tensor_util.constant_value(rank) for rank in ranks])
    if not any((r is None for r in ranks_static)):
        for rank_static in ranks_static:
            if rank_static.ndim != 0:
                raise ValueError('Rank must be a scalar.')
        x_rank_static = x.get_shape().ndims
        if x_rank_static is not None:
            if not static_condition(x_rank_static, ranks_static):
                raise ValueError('Static rank condition failed', x_rank_static, ranks_static)
            return control_flow_ops.no_op(name='static_checks_determined_all_ok')
    condition = dynamic_condition(array_ops.rank(x), ranks)
    for rank, rank_static in zip(ranks, ranks_static):
        if rank_static is None:
            this_data = ['Rank must be a scalar. Received rank: ', rank]
            rank_check = assert_rank(rank, 0, data=this_data)
            condition = control_flow_ops.with_dependencies([rank_check], condition)
    return control_flow_assert.Assert(condition, data, summarize=summarize)