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
@ops.RegisterGradient('Conv3D')
def _Conv3DGrad(op, grad):
    data_format = op.get_attr('data_format').decode()
    shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
    return [nn_ops.conv3d_backprop_input_v2(shape_0, op.inputs[1], grad, dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), data_format=data_format), nn_ops.conv3d_backprop_filter_v2(op.inputs[0], shape_1, grad, dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), data_format=data_format)]