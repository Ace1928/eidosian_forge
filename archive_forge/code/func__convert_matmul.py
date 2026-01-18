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
@RegisterPFor('MatMul')
def _convert_matmul(pfor_input):
    a, a_stacked, _ = pfor_input.input(0)
    b, b_stacked, _ = pfor_input.input(1)
    tr_a = pfor_input.get_attr('transpose_a')
    tr_b = pfor_input.get_attr('transpose_b')
    if a_stacked and b_stacked:
        output = wrap(math_ops.matmul(a, b, adjoint_a=tr_a, adjoint_b=tr_b), True)
        return output
    elif a_stacked:
        if tr_a:
            a = array_ops.transpose(a, [0, 2, 1])
        if a.shape.is_fully_defined():
            x, y, z = a.shape
        else:
            x, y, z = [array_ops.reshape(i, []) for i in array_ops.split(array_ops.shape(a), 3)]
        a = array_ops.reshape(a, [x * y, z])
        prod = math_ops.matmul(a, b, transpose_b=tr_b)
        return wrap(array_ops.reshape(prod, [x, y, -1]), True)
    else:
        assert b_stacked
        if tr_b:
            perm = [2, 0, 1]
            b = array_ops.transpose(b, perm)
        else:
            b_shape = array_ops.shape(b)
            min_dim = math_ops.minimum(b_shape[0], b_shape[1])
            perm = array_ops.where(math_ops.equal(min_dim, 1), [0, 1, 2], [1, 0, 2])
            new_shape = array_ops_stack.stack([b_shape[1], b_shape[0], b_shape[2]])
            b = array_ops.transpose(b, perm)
            b = array_ops.reshape(b, new_shape)
        if b.shape.is_fully_defined():
            x, y, z = b.shape
        else:
            x, y, z = [array_ops.reshape(i, []) for i in array_ops.split(array_ops.shape(b), 3)]
        b = array_ops.reshape(b, [x, y * z])
        prod = math_ops.matmul(a, b, transpose_a=tr_a)
        prod = array_ops.reshape(prod, [-1, y, z])
        prod = array_ops.transpose(prod, [1, 0, 2])
        return wrap(prod, True)