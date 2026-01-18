from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def _as_operation(op_or_tensor):
    if isinstance(op_or_tensor, tensor_lib.Tensor):
        return op_or_tensor.op
    return op_or_tensor