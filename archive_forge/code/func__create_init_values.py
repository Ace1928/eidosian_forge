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
def _create_init_values(self, pfor_input):
    """Create arguments passed to converted while_loop."""
    with ops.name_scope('while_init'):
        loop_len_vector = pfor_input.pfor.loop_len_vector
        loop_len = loop_len_vector[0]
        num_outputs = len(self._outputs)
        inputs = []
        maybe_stacked_cache = {}
        for i, enter in enumerate(self._enters):
            inp, stacked = self._convert_enter(pfor_input.pfor, enter)
            inputs.append(inp)
            maybe_stacked_cache[enter] = stacked
            if i < num_outputs:
                maybe_stacked_cache[self._exit_switches[i].outputs[1]] = stacked
        input_shape_invariants = []
        output_tas = []
        ta_shape_invariants = []
        inputs_stacked = []
        for i, inp in enumerate(inputs):
            enter = self._enters[i]
            inp_stacked = self._maybe_stacked(maybe_stacked_cache, enter)
            if i < num_outputs:
                body_output = self._body_outputs[i]
                if enter.op in self._pfor_ops:
                    body_output_stacked = self._maybe_stacked(maybe_stacked_cache, body_output)
                else:
                    body_output_stacked = False
                if body_output_stacked and (not inp_stacked):
                    inp = _stack(inp, loop_len_vector).t
                    inputs[i] = inp
                    inp_stacked = True
                output_tas.append(tensor_array_ops.TensorArray(inp.dtype, loop_len))
                ta_shape_invariants.append(tensor_shape.TensorShape(None))
            inputs_stacked.append(inp_stacked)
            input_shape_invariants.append(tensor_shape.TensorShape(None))
        init_values = [True, pfor_input.pfor.all_indices] + inputs + output_tas
        shape_invariants = [tensor_shape.TensorShape(None), tensor_shape.TensorShape(None)] + input_shape_invariants + ta_shape_invariants
        return (init_values, inputs_stacked, shape_invariants)