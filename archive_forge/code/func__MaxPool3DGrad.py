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
@ops.RegisterGradient('MaxPool3D')
def _MaxPool3DGrad(op, grad):
    return gen_nn_ops.max_pool3d_grad(op.inputs[0], op.outputs[0], grad, ksize=op.get_attr('ksize'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), data_format=op.get_attr('data_format').decode())