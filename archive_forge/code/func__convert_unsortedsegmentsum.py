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
@RegisterPForWithArgs('UnsortedSegmentSum', math_ops.unsorted_segment_sum)
@RegisterPForWithArgs('UnsortedSegmentMax', math_ops.unsorted_segment_max)
@RegisterPForWithArgs('UnsortedSegmentMin', math_ops.unsorted_segment_min)
@RegisterPForWithArgs('UnsortedSegmentProd', math_ops.unsorted_segment_prod)
def _convert_unsortedsegmentsum(pfor_input, _, op_func):
    pfor_input.stack_inputs([0, 1])
    data = pfor_input.stacked_input(0)
    segment_ids = pfor_input.stacked_input(1)
    num_segments = pfor_input.unstacked_input(2)
    if segment_ids.dtype != num_segments.dtype:
        segment_ids = math_ops.cast(segment_ids, dtypes.int64)
        num_segments = math_ops.cast(num_segments, dtypes.int64)
    dtype = segment_ids.dtype
    segment_shape = array_ops.shape(segment_ids, out_type=dtype)
    n = segment_shape[0]
    ones = array_ops.ones_like(segment_shape, dtype=dtype)[1:]
    segment_offset = num_segments * math_ops.range(n, dtype=dtype)
    segment_offset = array_ops.reshape(segment_offset, array_ops.concat([[n], ones], axis=0))
    segment_ids += segment_offset
    num_segments = math_ops.cast(num_segments, dtypes.int64) * math_ops.cast(n, dtypes.int64)
    output = op_func(data, segment_ids, num_segments)
    new_output_shape = array_ops.concat([[n, -1], array_ops.shape(output)[1:]], axis=0)
    output = array_ops.reshape(output, new_output_shape)
    return wrap(output, True)