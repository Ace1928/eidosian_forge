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
@ops.RegisterGradient('LeakyReluGrad')
def _LeakyReluGradGrad(op, grad):
    x = op.inputs[1]
    alpha = op.get_attr('alpha')
    return (gen_nn_ops.leaky_relu_grad(grad, x, alpha=alpha), array_ops.zeros_like(x))