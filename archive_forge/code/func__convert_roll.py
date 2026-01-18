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
@RegisterPFor('Roll')
def _convert_roll(pfor_input):
    t = pfor_input.stacked_input(0)
    shift, shift_stacked, _ = pfor_input.input(1)
    axis = pfor_input.unstacked_input(2)
    if not shift_stacked:
        return wrap(manip_ops.roll(t, shift, axis + 1), True)
    else:
        num_unstacked_axes = math_ops.cast(array_ops.rank(t), dtypes.int64) - 1
        axis = math_ops.cast(array_ops.reshape(axis, [-1]), dtypes.int64)
        loop_len = math_ops.cast(pfor_input.pfor.loop_len_vector[0], dtypes.int64)
        shift = math_ops.cast(array_ops.reshape(shift, [loop_len, -1]), dtypes.int64)
        axis_segment_ids = math_ops.range(loop_len, dtype=dtypes.int64)[:, None] * num_unstacked_axes + axis[None, :]
        axis_offsets = array_ops.reshape(math_ops.unsorted_segment_sum(data=shift, segment_ids=axis_segment_ids, num_segments=loop_len * num_unstacked_axes), [loop_len, num_unstacked_axes])
        unstacked_shape = array_ops.shape(t, out_type=dtypes.int64)[1:]
        cumsize = math_ops.cumprod(unstacked_shape, exclusive=True, reverse=True)
        num_unstacked_elements = math_ops.reduce_prod(unstacked_shape)
        result_coordinates = (math_ops.range(num_unstacked_elements, dtype=dtypes.int64)[None, :, None] // cumsize[None, None, :] - axis_offsets[:, None, :]) % unstacked_shape[None, None, :]
        result_flat = array_ops.gather_nd(params=t, indices=result_coordinates, batch_dims=1)
        return wrap(array_ops.reshape(result_flat, array_ops.shape(t)), True)