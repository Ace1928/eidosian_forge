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
@ops.RegisterGradient('MaxPool3DGrad')
def _MaxPool3DGradGrad(op, grad):
    return (array_ops.zeros_like(op.inputs[0]), array_ops.zeros_like(op.inputs[1]), gen_nn_ops.max_pool3d_grad_grad(op.inputs[0], op.inputs[1], grad, op.get_attr('ksize'), op.get_attr('strides'), padding=op.get_attr('padding'), data_format=op.get_attr('data_format').decode()))