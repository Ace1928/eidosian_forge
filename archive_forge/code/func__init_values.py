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
def _init_values(self):
    """Create arguments passed to converted while_loop."""
    loop_len = self._pfor.loop_len_vector[0]
    inputs = []
    output_tas = []
    with ops.name_scope('while_init'):
        for inp in self._pfor_input.inputs:
            inputs.append(inp.t)
            variant_type_id = _variant_type_id(inp.t)
            if variant_type_id in _INTERNAL_STACKING_TYPE_IDS:
                if variant_type_id != full_type_pb2.TFT_ARRAY:
                    raise NotImplementedError(f'While loop conversion is only supported for TensorLists. Got another variant {inp.t}, probably an optional. Please file a bug.')
                element_shape = list_ops.tensor_list_element_shape(inp.t, dtypes.int32)
                if inp.is_stacked:
                    element_shape = tf_cond.cond(math_ops.equal(array_ops.rank(element_shape), 0), lambda: element_shape, lambda: element_shape[1:])
                dtype = _parse_variant_shapes_and_types(inp.t)[0].dtype

                def _init_loop_body(index, output_ta):
                    output_ta = output_ta.write(index, list_ops.tensor_list_reserve(element_shape, loop_len, dtype))
                    return (index + 1, output_ta)
                length = list_ops.tensor_list_length(inp.t)
                output_ta = tensor_array_ops.TensorArray(inp.t.dtype, size=length, dynamic_size=True, infer_shape=False)
                _, output_ta = while_loop.while_loop(lambda index, _: index < length, _init_loop_body, [0, output_ta])
            else:
                output_ta = tensor_array_ops.TensorArray(inp.t.dtype, size=loop_len, dynamic_size=False, infer_shape=True)
            output_tas.append(output_ta)
    indices = math_ops.range(self._pfor.loop_len_vector[0]) if self._pfor.all_indices_partitioned else self._pfor.all_indices
    return [True, indices] + inputs + output_tas