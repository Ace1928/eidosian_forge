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
@RegisterPFor('DepthwiseConv2dNativeBackpropFilter')
def _convert_depthwise_conv2d_native_backprop_filter(pfor_input):
    stack_size = pfor_input.pfor.loop_len_vector[0]
    data_format = pfor_input.get_attr('data_format')
    c_dim = 1 if data_format == b'NCHW' else 3
    inputs = _flatten_with_inner_dim(pfor_input.stacked_input(0), c_dim + 1, 5)
    filter_sizes = pfor_input.unstacked_input(1)
    filter_sizes_multipliers = [constant_op.constant([1, 1], dtype=dtypes.int32), [stack_size], constant_op.constant([1], dtype=dtypes.int32)]
    filter_sizes *= array_ops.concat(filter_sizes_multipliers, axis=0)
    out_backprop = _flatten_with_inner_dim(pfor_input.stacked_input(2), c_dim + 1, 5)
    result = _create_op('DepthwiseConv2dNativeBackpropFilter', [inputs, filter_sizes, out_backprop], [x.dtype for x in pfor_input.outputs], attrs=pfor_input.op.node_def.attr).outputs[0]
    return wrap(_unflatten_with_inner_dim(result, 2, 4, stack_size), True)