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
@ops.RegisterGradient('AvgPool')
def _AvgPoolGrad(op, grad):
    return gen_nn_ops.avg_pool_grad(array_ops.shape(op.inputs[0]), grad, op.get_attr('ksize'), op.get_attr('strides'), op.get_attr('padding'), data_format=op.get_attr('data_format'))