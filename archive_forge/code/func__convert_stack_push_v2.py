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
@RegisterPFor('StackPushV2')
def _convert_stack_push_v2(pfor_input):
    handle = pfor_input.unstacked_input(0)
    elem, elem_stacked, _ = pfor_input.input(1)
    swap_memory = pfor_input.get_attr('swap_memory')
    if not _stack_handle_inside_pfor(pfor_input.op.inputs[0], pfor_input):
        raise ValueError('StackPushV2 not allowed on stacks created outside pfor.')
    stack_cache_key = _stack_cache_key(pfor_input)
    stacked = _stack_cache.get(stack_cache_key, None)
    if stacked is None:
        stacked = elem_stacked
        _stack_cache[stack_cache_key] = stacked
    else:
        if not stacked and elem_stacked:
            raise ValueError('It looks like the stack was previously determined to be loop invariant, but we are now trying to push a loop dependent value to it. This is currently unsupported.')
        if stacked and (not elem_stacked):
            elem = _stack(elem, pfor_input.pfor.loop_len_vector).t
    out = data_flow_ops.stack_push_v2(handle, elem, swap_memory=swap_memory)
    return wrap(out, stacked)