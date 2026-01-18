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
def _process_body(self, inputs_stacked, new_indices, cond_stacked, new_inputs, not_all_done):
    """Convert the body function."""
    mismatching_stacked_indices = []

    def true_fn():
        """Converts the body function for all but last iteration."""
        wrapped_inputs = [wrap(inp, stacked) for inp, stacked in zip(new_inputs, inputs_stacked)]
        while True:
            if self._pfor.all_indices_partitioned:
                indices = array_ops.gather(self._pfor.all_indices, new_indices)
            else:
                indices = new_indices
            body_pfor = PFor(loop_var=self._pfor.loop_var, loop_len=array_ops.size(new_indices), pfor_ops=self._body_func.graph.get_operations(), fallback_to_while_loop=self._pfor.fallback_to_while_loop, all_indices=indices, all_indices_partitioned=self._pfor.all_indices_partitioned or cond_stacked, pfor_config=self._pfor.pfor_config)
            stacking_mismatch = False
            outputs = _convert_function_call(self._body_func, body_pfor, wrapped_inputs)
            for i, (out, inp) in enumerate(zip(outputs, wrapped_inputs)):
                if out.is_stacked != inp.is_stacked:
                    stacking_mismatch = True
                    mismatching_stacked_indices.append(i)
                    stacked = _stack(inp.t, [array_ops.size(new_indices)])
                    if inp.t.dtype == dtypes.variant:
                        stacked = wrap(_tile_variant_with_length(stacked.t, [array_ops.size(new_indices)]))
                    wrapped_inputs[i] = stacked
            if not stacking_mismatch:
                if mismatching_stacked_indices:
                    with ops.control_dependencies([control_flow_assert.Assert(False, ['pfor ERROR: this branch should never execute'])]):
                        return [array_ops.identity(x) for x in new_inputs]
                else:
                    return [out.t for out in outputs]
    return (tf_cond.cond(not_all_done, true_fn, lambda: list(new_inputs)), mismatching_stacked_indices)