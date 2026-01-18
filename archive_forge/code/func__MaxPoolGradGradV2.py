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
@ops.RegisterGradient('MaxPoolGradV2')
def _MaxPoolGradGradV2(op, grad):
    ksize = op.inputs[3]
    strides = op.inputs[4]
    return (array_ops.zeros_like(op.inputs[0]), array_ops.zeros_like(op.inputs[1]), gen_nn_ops.max_pool_grad_grad_v2(op.inputs[0], op.inputs[1], grad, ksize, strides, padding=op.get_attr('padding'), data_format=op.get_attr('data_format')), None, None)