import typing
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _copy_row_shape(rt_inputs, splits):
    """Sets splits.shape to [rt[shape[0]+1] for each rt in rt_inputs."""
    for rt in rt_inputs:
        if rt.shape[0] is not None:
            splits.set_shape(tensor_shape.TensorShape(rt.shape[0] + 1))