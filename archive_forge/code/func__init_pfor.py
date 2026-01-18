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
def _init_pfor(self, parent_pfor, indices, cond_stacked, inputs, inputs_stacked):
    """Create a PFor object for converting parts of the while_loop.

    Args:
      parent_pfor: PFor object being used for converting the while_loop.
      indices: int32 Tensor of ids for the iterations that are still active
        (i.e. did not exit the while_loop).
      cond_stacked: True if the while_loop condition is stacked.
      inputs: list of input Tensors corresponding 1-to-1 with self._enters. Note
        that these Tensors are a subset of the loop variables for the generated
        while_loop.
      inputs_stacked: List of booleans corresponding 1-to-1 with `inputs`,
        indicating if the value is stacked or not.

    Returns:
      A PFor instance. The instance is initialized by adding conversion mappings
        of nodes that will be external to the conversion that the returned
        instance will be used for. e.g. Enter nodes as well as Merge and Switch
        outputs are mapped to converted values.
    """
    num_outputs = len(self._outputs)
    assert len(inputs) == len(self._enters)
    assert len(inputs_stacked) == len(self._enters)
    loop_var = parent_pfor.loop_var
    loop_len = array_ops.size(indices)
    pfor = PFor(loop_var, loop_len, pfor_ops=self._pfor_ops, all_indices=indices, all_indices_partitioned=cond_stacked, fallback_to_while_loop=self._fallback_to_while_loop, pfor_config=self._pfor_config)
    for enter in self._direct_enters:
        enter_input = enter.op.inputs[0]
        converted_enter, stacked, is_sparse_stacked = parent_pfor._convert_helper(enter_input)
        assert not stacked and (not is_sparse_stacked), (enter, converted_enter)
        pfor._add_conversion(enter, wrap(converted_enter, False))
    for enter, inp, stacked in zip(self._enters, inputs, inputs_stacked):
        pfor._add_conversion(enter, wrap(inp, stacked))
    for i in range(num_outputs):
        wrapped_inp = wrap(inputs[i], inputs_stacked[i])
        merge = self._enter_merges[i]
        pfor._add_conversion(merge.outputs[0], wrapped_inp)
        pfor._add_conversion(merge.outputs[1], wrap(constant_op.constant(-1.0), False))
        switch = self._exit_switches[i]
        pfor._add_conversion(switch.outputs[1], wrapped_inp)
    return pfor