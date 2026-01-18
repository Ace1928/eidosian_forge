from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_utils
@ops.RegisterGradient('XlaSpmdFullToShardShape')
def _spmd_full_to_shard_shape_grad(op, grad):
    s2f = gen_xla_ops.xla_spmd_shard_to_full_shape(grad, manual_sharding=op.get_attr('manual_sharding'), full_shape=op.inputs[0].shape.as_list(), dim=op.get_attr('dim'), unspecified_dims=op.get_attr('unspecified_dims'))
    return [s2f]