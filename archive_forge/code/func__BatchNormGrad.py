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
def _BatchNormGrad(grad_y, x, scale, pop_mean, pop_var, epsilon, data_format, is_training=True):
    """Returns the gradients for the 3 inputs of BatchNorm.

  Args:
    grad_y: A `Tensor` of 4 or 5 dimensions for gradient for y.
    x: A `Tensor` of 4 or 5 dimensions for x.
    scale: A `Tensor` of 1 dimension for scaling.
    pop_mean: A `Tensor` of 1 dimension for the population mean. Only used when
      is_training=False.
    pop_var: A `Tensor` of 1 dimension for the population variance. Only used
      when is_training=False.
    epsilon: A small float number added to the variance of x.
    data_format: The data format for input. Either b"NHWC" or b"NCHW".
    is_training: A bool value to indicate the operation is for training
      (default) or inference.

  Returns:
    A tuple (grad_x, grad_scale, grad_offset), where grad_x is the gradient
    for x, grad_scale the gradient for scale, and grad_offset the gradient
    for offset.
  """
    x_dtype = x.dtype.base_dtype
    if x_dtype == dtypes.float16 or x_dtype == dtypes.bfloat16:
        x = math_ops.cast(x, dtypes.float32)
        grad_y = math_ops.cast(grad_y, dtypes.float32)
    if is_training:
        if data_format == b'NHWC':
            keepdims = False
            reduce_axis = [0, 1, 2]
        elif data_format == b'NDHWC':
            keepdims = False
            reduce_axis = [0, 1, 2, 3]
        elif data_format == b'NCHW':
            keepdims = True
            reduce_axis = [0, 2, 3]
            shape = [1, array_ops.size(scale), 1, 1]
            scale = array_ops.reshape(scale, shape)
        else:
            keepdims = True
            reduce_axis = [0, 2, 3, 4]
            shape = [1, array_ops.size(scale), 1, 1, 1]
            scale = array_ops.reshape(scale, shape)
        mean_grad_y = math_ops.reduce_mean(grad_y, reduce_axis, keepdims=keepdims)
        mean_x = math_ops.reduce_mean(x, reduce_axis, keepdims=keepdims)
        var_x = math_ops.reduce_mean(math_ops.squared_difference(x, array_ops.stop_gradient(mean_x)), reduce_axis, keepdims=keepdims)
        grad_y_offset = grad_y - mean_grad_y
        x_offset = x - mean_x
        mean = math_ops.reduce_mean(grad_y * x_offset, axis=reduce_axis, keepdims=keepdims)
        grad_x = scale * math_ops.rsqrt(var_x + epsilon) * (grad_y_offset - math_ops.reciprocal(var_x + epsilon) * mean * x_offset)
        grad_scale = math_ops.rsqrt(var_x + epsilon) * math_ops.reduce_sum(grad_y * x_offset, axis=reduce_axis, keepdims=keepdims)
        if data_format == b'NCHW' or data_format == b'NCDHW':
            grad_scale = array_ops.squeeze(grad_scale)
        grad_offset = math_ops.reduce_sum(grad_y, axis=reduce_axis)
        return (math_ops.cast(grad_x, x_dtype), grad_scale, grad_offset)
    else:
        if data_format == b'NHWC':
            reduce_axis = [0, 1, 2]
        elif data_format == b'NDHWC':
            reduce_axis = [0, 1, 2, 3]
        elif data_format == b'NCHW':
            reduce_axis = [0, 2, 3]
            shape = [1, array_ops.size(pop_mean), 1, 1]
            pop_mean = array_ops.reshape(pop_mean, shape)
            pop_var = array_ops.reshape(pop_var, shape)
            scale = array_ops.reshape(scale, shape)
        else:
            reduce_axis = [0, 2, 3, 4]
            shape = [1, array_ops.size(pop_mean), 1, 1, 1]
            pop_mean = array_ops.reshape(pop_mean, shape)
            pop_var = array_ops.reshape(pop_var, shape)
            scale = array_ops.reshape(scale, shape)
        grad_offset = math_ops.reduce_sum(grad_y, axis=reduce_axis)
        var_rsqrt = math_ops.rsqrt(pop_var + epsilon)
        grad_scale = math_ops.reduce_sum(grad_y * (x - pop_mean) * var_rsqrt, axis=reduce_axis)
        grad_x = grad_y * scale * var_rsqrt
        return (math_ops.cast(grad_x, x_dtype), grad_scale, grad_offset)