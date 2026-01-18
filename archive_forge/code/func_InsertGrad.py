from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_map_ops
from tensorflow.python.ops.gen_map_ops import *
@ops.RegisterGradient('TensorMapInsert')
def InsertGrad(op, dmap):
    _, k, v = op.inputs
    key_grad = None
    value_grad, map_grad = cond.cond(tensor_map_has_key(dmap, k), lambda: (tensor_map_lookup(dmap, k, v.dtype), tensor_map_erase(dmap, k, v.dtype)), lambda: (array_ops.zeros_like(v), dmap))
    return (map_grad, key_grad, value_grad)