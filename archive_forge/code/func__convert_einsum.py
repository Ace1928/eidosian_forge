import collections
from functools import partial
import string
import sys
import traceback
import numpy as np
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
@RegisterPForWithArgs('XlaEinsum')
@RegisterPForWithArgs('Einsum')
def _convert_einsum(pfor_input, op_type):
    inputs, input_stacked, _ = zip(*[pfor_input.input(i) for i in range(pfor_input.num_inputs)])
    equation = pfor_input.get_attr('equation').decode('utf-8')
    input_expr, output_expr = equation.split('->')
    input_exprs = input_expr.split(',')
    chosen_symbol = None
    for s in string.ascii_letters:
        if s in equation:
            continue
        else:
            chosen_symbol = s
            break
    if chosen_symbol is None:
        raise ValueError('Could not figure out what symbol to use for new axis.')
    assert any(input_stacked)
    for i in range(len(inputs)):
        if input_stacked[i]:
            input_exprs[i] = '{}{}'.format(chosen_symbol, input_exprs[i])
    output_expr = '{}{}'.format(chosen_symbol, output_expr)
    new_equation = '{}->{}'.format(','.join(input_exprs), output_expr)
    if op_type == 'XlaEinsum':
        if len(inputs) == 1:
            result = xla.einsum(equation=new_equation, a=inputs[0])
        else:
            result = xla.einsum(equation=new_equation, a=inputs[0], b=inputs[1])
    else:
        assert op_type == 'Einsum'
        result = special_math_ops.einsum(new_equation, *inputs)
    return wrap(result, True)