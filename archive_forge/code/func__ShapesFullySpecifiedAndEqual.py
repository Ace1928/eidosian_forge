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
def _ShapesFullySpecifiedAndEqual(x, y, grad):
    x_shape = x._shape_tuple()
    y_shape = y._shape_tuple()
    grad_shape = grad._shape_tuple()
    return x_shape == y_shape and x_shape == grad_shape and (x_shape is not None) and (None not in x_shape)