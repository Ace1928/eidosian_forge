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
def _einsum_v1(equation, *inputs, **kwargs):
    """Legacy implementation of einsum without using EinsumOp."""
    name = kwargs.pop('name', None)
    if kwargs:
        raise TypeError(f'Invalid keyword arguments for this function: {', '.join([format(key) for key in sorted(list(kwargs.keys()))])}. Expected: name.')
    with ops.name_scope(name, 'einsum', [equation, inputs]) as name:
        inputs = list(inputs)
        input_shapes = [x.shape for x in inputs]
        input_axis_labels, output_axis_labels = _einsum_v1_parse_and_resolve_equation(equation, input_shapes)
        axis_labels = set(''.join(input_axis_labels) + output_axis_labels)
        for a in axis_labels:
            for input_labels in input_axis_labels:
                if len(input_axis_labels) == 1 and input_labels.count(a) == 2 and (input_labels == input_labels[::-1]) and ('->' not in equation):
                    return math_ops.trace(inputs[0])
                if input_labels.count(a) > 1:
                    raise ValueError(f'Subscript not supported: the axis {a} appears more than once in {input_labels}.')
        for a in axis_labels:
            input_count = sum((1 for s in input_axis_labels if a in s))
            if input_count > 2 and a not in output_axis_labels:
                logging.warn(f'Falling back to exponential-space implementation of einsum() because index {a} is summed over more than two inputs.')
                return _exponential_space_einsum_v1(equation, *inputs)
        if _enclosing_tpu_context() is not None and len(inputs) == 2:
            return gen_xla_ops.xla_einsum(inputs[0], inputs[1], input_axis_labels[0] + ',' + input_axis_labels[1] + '->' + output_axis_labels)
        temp = inputs[0]
        temp_axis_labels = input_axis_labels[0]
        for i in range(len(inputs) - 1):
            axes_to_sum = set(temp_axis_labels) & set(input_axis_labels[i + 1]) - set(output_axis_labels)
            temp, temp_axis_labels = _einsum_v1_reduction(temp, temp_axis_labels, inputs[i + 1], input_axis_labels[i + 1], axes_to_sum)
        missing_indices = set(temp_axis_labels) - set(output_axis_labels)
        if missing_indices:
            axis = [i for i, a in enumerate(temp_axis_labels) if a not in output_axis_labels]
            temp = math_ops.reduce_sum(temp, axis=axis)
            temp_axis_labels = ''.join((a for a in temp_axis_labels if a in output_axis_labels))
        if sorted(temp_axis_labels) != sorted(output_axis_labels):
            raise ValueError(f'Invalid equation: {equation}. The computed and specified output labels do not match: {temp_axis_labels} vs {output_axis_labels}.')
        perm = [temp_axis_labels.index(a) for a in output_axis_labels]
        return _transpose_if_necessary(temp, perm)