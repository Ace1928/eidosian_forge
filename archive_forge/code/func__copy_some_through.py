from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _copy_some_through(current, candidate):
    """Copy some tensors through via array_ops.where."""

    def copy_fn(cur_i, cand_i):
        if isinstance(cur_i, tensor_array_ops.TensorArray):
            return cand_i
        if cur_i.shape.rank == 0:
            return cand_i
        with ops.colocate_with(cand_i):
            return array_ops.where(elements_finished, cur_i, cand_i)
    return nest.map_structure(copy_fn, current, candidate)