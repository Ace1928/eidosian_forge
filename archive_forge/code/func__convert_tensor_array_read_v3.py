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
@RegisterPFor('TensorArrayReadV3')
def _convert_tensor_array_read_v3(pfor_input):
    handle = pfor_input.unstacked_input(0)
    index, index_stacked, _ = pfor_input.input(1)
    dtype = pfor_input.get_attr('dtype')
    flow, flow_stacked, _ = pfor_input.input(2)
    if flow_stacked:
        flow = _unstack_flow(flow)
    is_inside_pfor = _handle_inside_pfor(pfor_input, pfor_input.op.inputs[0])
    if is_inside_pfor:
        all_indices = pfor_input.pfor.all_indices
        all_indices_partitioned = pfor_input.pfor.all_indices_partitioned
        if index_stacked:
            if flow_stacked:
                raise ValueError('It looks like TensorArrayReadV3 was called on a TensorArray whose values are not loop-invariant, and the read indices were also not loop invariant. This is currently unsupported.')
            value = data_flow_ops.tensor_array_gather_v3(handle, index, flow, dtype=dtype)
            return wrap(value, True)
        value = data_flow_ops.tensor_array_read_v3(handle, index, flow, dtype=dtype)
        if flow_stacked and all_indices_partitioned:
            value = array_ops.gather(value, all_indices)
        return wrap(value, flow_stacked)
    if index_stacked:
        value = data_flow_ops.tensor_array_gather_v3(handle, index, flow, dtype=dtype)
    else:
        value = data_flow_ops.tensor_array_read_v3(handle, index, flow, dtype=dtype)
    return wrap(value, index_stacked)