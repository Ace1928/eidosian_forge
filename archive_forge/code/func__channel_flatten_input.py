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
def _channel_flatten_input(x, data_format):
    """Merge the stack dimension with the channel dimension.

  If S is pfor's stacking dimension, then,
    - for SNCHW, we transpose to NSCHW. If N dimension has size 1, the transpose
      should be cheap.
    - for SNHWC, we transpose to NHWSC.
  We then merge the S and C dimension.

  Args:
    x: tensor_lib.Tensor to transform.
    data_format: "NCHW" or "NHWC".

  Returns:
    A 3-element tuple with the transformed value, along with the shape for
    reshape and order for transpose required to transform back.
  """
    graph = ops.get_default_graph()
    cache_key = (graph, x.ref(), data_format)
    if cache_key not in _channel_flatten_input_cache:
        x_shape = array_ops.shape(x)
        neg_ones = constant_op.constant([-1], dtype=x_shape.dtype)
        if data_format == b'NCHW':
            order = [1, 0, 2, 3, 4]
            shape = array_ops.concat([x_shape[1:2], neg_ones, x_shape[3:]], axis=0)
            reverse_order = order
        else:
            order = [1, 2, 3, 0, 4]
            shape = array_ops.concat([x_shape[1:4], neg_ones], axis=0)
            reverse_order = [3, 0, 1, 2, 4]
        x = array_ops.transpose(x, order)
        reverse_shape = array_ops.shape(x)
        x = array_ops.reshape(x, shape)
        outputs = (x, reverse_order, reverse_shape)
        _channel_flatten_input_cache[cache_key] = outputs
    else:
        outputs = _channel_flatten_input_cache[cache_key]
    return outputs