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
@ops.RegisterGradient('SoftplusGrad')
def _SoftplusGradGrad(op, grad):
    dy, x = op.inputs
    with ops.control_dependencies([grad]):
        ddy = gen_nn_ops.softplus_grad(grad, x)
        d2x = grad * dy / (math_ops.exp(-x) + 2.0 + math_ops.exp(x))
        return (ddy, d2x)