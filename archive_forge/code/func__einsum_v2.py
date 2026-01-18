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
def _einsum_v2(equation, *inputs, **kwargs):
    """Implementation of einsum utilizing opt_einsum and EinsumOp."""
    name = kwargs.pop('name', None)
    optimize = kwargs.pop('optimize', 'greedy')
    if kwargs:
        raise TypeError(f'Invalid keyword arguments for einsum: {', '.join(kwargs)}. Valid arguments: name, optimize, greedy.')
    with ops.name_scope(name, 'einsum', [equation, inputs]) as name:
        inputs = list(inputs)
        input_shapes = []
        for operand in inputs:
            if isinstance(operand.shape, tensor_shape.TensorShape):
                input_shapes.append(operand.shape.as_list() if operand.shape else None)
            else:
                input_shapes.append(list(operand.shape))
        resolved_equation, resolved_input_shapes, ellipsis_label = _einsum_v2_parse_and_resolve_equation(equation, input_shapes)
        if len(inputs) <= 2:
            if ellipsis_label:
                resolved_equation = resolved_equation.replace(ellipsis_label, '...')
            return gen_linalg_ops.einsum(inputs, resolved_equation)
        shaped = collections.namedtuple('shaped', ['shape'])
        shaped_inputs = tuple([shaped(tuple(shape)) for shape in resolved_input_shapes])
        indices_and_equations = _get_opt_einsum_contract_path(resolved_equation, shaped_inputs, optimize)
        for operand_indices, binary_equation in indices_and_equations:
            if ellipsis_label:
                binary_equation = binary_equation.replace(ellipsis_label, '...')
            operands = list(map(inputs.pop, operand_indices))
            inputs.append(gen_linalg_ops.einsum(operands, binary_equation))
        return inputs[0]