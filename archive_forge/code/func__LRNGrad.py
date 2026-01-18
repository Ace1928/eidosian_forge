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
@ops.RegisterGradient('LRN')
def _LRNGrad(op, grad):
    depth_radius = op.get_attr('depth_radius')
    bias = op.get_attr('bias')
    alpha = op.get_attr('alpha')
    beta = op.get_attr('beta')
    return [gen_nn_ops.lrn_grad(grad, op.inputs[0], op.outputs[0], depth_radius, bias, alpha, beta)]