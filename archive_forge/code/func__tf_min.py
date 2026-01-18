import inspect
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
def _tf_min(*args, **kwargs):
    if len(kwargs):
        kwargs_tuple = tuple(set(kwargs.keys()))
        raise ValueError('These keyword arguments are currently not supported: {}'.format(kwargs_tuple))
    if len(args) == 1:
        rank = args[0].shape.rank
        if rank == 0:
            return args[0]
        if rank == 1:
            return math_ops.reduce_min(*args, axis=0)
        raise ValueError('min(arg) currently support only tensor with rank 1, but got a tensor with rank {}'.format(rank))
    for arg in args:
        rank = arg.shape.rank
        if rank != 0:
            raise ValueError('min(arg1, arg2, *args) currently support only scalar tensor, but got a tensor with shape {}'.format(rank))
    return math_ops.reduce_min(args, axis=0)