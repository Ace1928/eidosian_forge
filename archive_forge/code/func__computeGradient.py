import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def _computeGradient(self, np_input, bias, dtype, data_format):
    input_shape = output_shape = np_input.shape
    bias_shape = bias.shape
    input_tensor = constant_op.constant(np_input, shape=input_shape, dtype=dtype)
    bias_tensor = constant_op.constant(bias, shape=bias_shape, dtype=dtype)
    if context.executing_eagerly():

        def bias_add(input_tensor, bias_tensor):
            return nn_ops.bias_add(input_tensor, bias_tensor, data_format=data_format)

        def bias_add_1(input_tensor):
            return bias_add(input_tensor, bias_tensor)

        def bias_add_2(bias_tensor):
            return bias_add(input_tensor, bias_tensor)
        input_jacob_a, input_jacob_n = gradient_checker_v2.compute_gradient(bias_add_1, [input_tensor])
        bias_jacob_a, bias_jacob_n = gradient_checker_v2.compute_gradient(bias_add_2, [bias_tensor])

        def bias_add_grad_function(upstream_gradients):
            with backprop.GradientTape() as tape:
                tape.watch(bias_tensor)
                bias_add_output = bias_add(input_tensor, bias_tensor)
                gradient_injector_output = bias_add_output * upstream_gradients
                return tape.gradient(gradient_injector_output, bias_tensor)
        upstream_tensor = self._random_tensor(output_shape, dtype)
        grad_jacob_a, grad_jacob_n = gradient_checker_v2.compute_gradient(bias_add_grad_function, [upstream_tensor])
    else:
        output_tensor = nn_ops.bias_add(input_tensor, bias_tensor, data_format=data_format)
        jacobians = gradient_checker.compute_gradient([input_tensor, bias_tensor], [input_shape, bias_shape], output_tensor, output_shape)
        (input_jacob_a, input_jacob_n), (bias_jacob_a, bias_jacob_n) = jacobians
        if dtype == dtypes.bfloat16:
            output_tensor = math_ops.cast(output_tensor, dtype=dtypes.float32)
        bias_add_grad = gradients_impl.gradients(nn_ops.l2_loss(output_tensor), bias_tensor)[0]
        grad_jacob_a, grad_jacob_n = gradient_checker.compute_gradient(output_tensor, output_shape, bias_add_grad, bias_shape)
    return ((input_jacob_a, bias_jacob_a, grad_jacob_a), (input_jacob_n, bias_jacob_n, grad_jacob_n))