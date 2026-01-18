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
@ops.RegisterGradient('DepthwiseConv2dNativeBackpropFilter')
def _DepthwiseConv2dNativeBackpropFilterGrad(op, grad):
    return [gen_nn_ops.depthwise_conv2d_native_backprop_input(array_ops.shape(op.inputs[0]), grad, op.inputs[2], dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), explicit_paddings=op.get_attr('explicit_paddings'), data_format=op.get_attr('data_format')), None, gen_nn_ops.depthwise_conv2d_native(op.inputs[0], grad, dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), explicit_paddings=op.get_attr('explicit_paddings'), data_format=op.get_attr('data_format'))]