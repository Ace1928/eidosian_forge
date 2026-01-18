from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_map_ops
from tensorflow.python.ops.gen_map_ops import *
@ops.RegisterGradient('TensorMapLookup')
def LookupGrad(op, dval):
    _, k = op.inputs
    map_grad = empty_tensor_map()
    map_grad = tensor_map_insert(map_grad, k, dval)
    key_grad = None
    return (map_grad, key_grad)