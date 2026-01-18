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
@ops.RegisterGradient('FusedBatchNormGradV3')
def _FusedBatchNormGradGradV3(op, *grad):
    grad_grad_y, grad_x, grad_scale, _, _ = _FusedBatchNormGradGrad(op, *grad)
    return (grad_grad_y, grad_x, grad_scale, None, None, None)