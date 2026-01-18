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
@ops.RegisterGradient('Dropout')
def _DropoutGrad(op, grad, _):
    dx = gen_nn_ops.dropout_grad(grad, op.inputs[1], op.outputs[1])
    return [dx, None, None, None, None]