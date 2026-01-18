import functools
import itertools
import operator
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
def GetLiftedIdx(d):
    if d == reduction_dim:
        return array_ops.reshape(op.outputs[1], lifted_idx_shape)
    iota_len = idx_shape[d]
    iota_shape = list(itertools.repeat(1, rank + 1))
    iota_shape[d] = iota_len
    iota = array_ops.reshape(math_ops.range(iota_len), iota_shape)
    return array_ops.broadcast_to(iota, lifted_idx_shape)