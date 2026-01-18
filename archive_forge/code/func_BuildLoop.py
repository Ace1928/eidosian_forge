import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def BuildLoop(self, pred, body, loop_vars, shape_invariants, return_same_structure):
    """Add the loop termination condition and body to the graph."""
    flat_orig_loop_vars = nest.flatten(loop_vars, expand_composites=True)
    loop_vars = nest.map_structure(_convert_to_tensor_or_composite_or_tensorarray, loop_vars)
    flat_loop_vars = nest.map_structure(_convert_tensorarray_to_flow, nest.flatten(loop_vars, expand_composites=True))
    if shape_invariants is not None:
        loop_vars_signature = nest.map_structure(_shape_invariant_to_type_spec, loop_vars, shape_invariants)
    else:
        loop_vars_signature = nest.map_structure(_shape_invariant_to_type_spec, loop_vars)
    try:
        self.Enter()
        with ops.get_default_graph()._mutation_lock():
            original_body_result, exit_vars = self._BuildLoop(pred, body, flat_orig_loop_vars, flat_loop_vars, loop_vars_signature)
    finally:
        self.Exit()
    flat_result = nest.flatten(original_body_result, expand_composites=True)
    exit_vars_with_tensorarrays = nest.map_structure(_convert_flow_to_tensorarray, flat_result, exit_vars)
    packed_exit_vars = nest.pack_sequence_as(structure=original_body_result, flat_sequence=exit_vars_with_tensorarrays, expand_composites=True)
    if return_same_structure:
        return packed_exit_vars
    else:
        return packed_exit_vars[0] if len(exit_vars) == 1 else packed_exit_vars