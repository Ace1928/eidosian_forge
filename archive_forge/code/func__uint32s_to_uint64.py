import enum
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
def _uint32s_to_uint64(x):
    return bitwise_ops.bitwise_or(math_ops.cast(x[0], dtypes.uint64), bitwise_ops.left_shift(math_ops.cast(x[1], dtypes.uint64), constant_op.constant(32, dtypes.uint64)))