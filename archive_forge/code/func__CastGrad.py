import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
@ops.RegisterGradient('Cast')
def _CastGrad(op, grad):
    t = [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16, dtypes.complex64, dtypes.complex128]
    src_type = op.inputs[0].dtype.base_dtype
    dst_type = grad.dtype.base_dtype
    if src_type in t and dst_type in t:
        return math_ops.cast(grad, src_type)
    else:
        return None