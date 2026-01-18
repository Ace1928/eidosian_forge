import collections
import functools
import re
import string
import numpy as np
import opt_einsum
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@ops.RegisterGradient('XlaEinsum')
def _einsum_grad(op, grad):
    equation = op.get_attr('equation')
    if isinstance(equation, bytes):
        equation = equation.decode()
    inputs, output = equation.split('->')
    left, right = inputs.split(',')
    return [gen_xla_ops.xla_einsum(grad, op.inputs[1], equation='{},{}->{}'.format(output, right, left), name=None), gen_xla_ops.xla_einsum(grad, op.inputs[0], equation='{},{}->{}'.format(output, left, right), name=None)]