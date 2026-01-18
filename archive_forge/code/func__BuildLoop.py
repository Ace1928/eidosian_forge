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
def _BuildLoop(self, pred, body, flat_orig_loop_vars, flat_loop_vars, loop_vars_signature):
    """Core: Add the loop termination condition and body to the graph."""
    flat_shape_invariants = nest.map_structure(lambda spec: spec.shape, nest.flatten(loop_vars_signature, expand_composites=True))
    self._InitializeValues(flat_loop_vars)
    if self._outer_context:
        real_vars = [self._outer_context.AddValue(x) for x in flat_loop_vars]
    else:
        real_vars = flat_loop_vars
    enter_vars = []
    with ops.control_dependencies(None):
        for real_var, shape_invariant in zip(real_vars, flat_shape_invariants):
            enter_var = _Enter(real_var, self._name, is_constant=False, parallel_iterations=self._parallel_iterations, use_input_shape=False)
            if _ShapeLessThanOrEqual(real_var.get_shape(), shape_invariant):
                enter_var.set_shape(shape_invariant)
            else:
                raise ValueError(f'The shape invariant specified for {real_var.name} is not compatible with the initial shape of the loop variable. It enters the loop with shape {real_var.get_shape()}, but the specified shape invariant is {shape_invariant}.')
            enter_var.graph.prevent_feeding(enter_var)
            if self._outer_context:
                self._outer_context.AddInnerOp(enter_var.op)
            enter_vars.append(enter_var)
    outer_context = self._outer_context
    control_pivot = None
    while outer_context is not None and control_pivot is None:
        control_pivot = outer_context.GetControlPivot()
        outer_context = outer_context._outer_context
    if control_pivot is not None:
        for var in enter_vars:
            if util.IsLoopConstantEnter(var.op.inputs[0].op):
                var.op._add_control_input(control_pivot.op)
    self._FixControlInputsAndContext(enter_vars)
    self._InitializeValues(enter_vars)
    self._loop_enters = enter_vars
    merge_vars = [merge([x, x])[0] for x in enter_vars]
    self._pivot_for_pred = merge_vars[0]
    merge_vars_with_tensorarrays = nest.map_structure(_convert_flow_to_tensorarray, flat_orig_loop_vars, merge_vars)
    packed_vars = nest.pack_sequence_as(structure=loop_vars_signature, flat_sequence=merge_vars_with_tensorarrays, expand_composites=True)
    c = ops.convert_to_tensor(pred(*packed_vars))
    self._pivot = loop_cond(c, name='LoopCond')
    switch_vars = [_SwitchRefOrTensor(x, self._pivot) for x in merge_vars]
    vars_for_body = [_Identity(x[1]) for x in switch_vars]
    self._pivot_for_body = vars_for_body[0]
    vars_for_body_with_tensorarrays = nest.map_structure(_convert_flow_to_tensorarray, flat_orig_loop_vars, vars_for_body)
    packed_vars_for_body = nest.pack_sequence_as(structure=loop_vars_signature, flat_sequence=vars_for_body_with_tensorarrays, expand_composites=True)
    pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)
    body_result = body(*packed_vars_for_body)
    post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)
    if not nest.is_nested(body_result):
        body_result = [body_result]
    if len(post_summaries) > len(pre_summaries):
        new_summaries = post_summaries[len(pre_summaries):]
        summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)
        summary_ref[:] = pre_summaries
        with ops.control_dependencies(new_summaries):

            def map_fn(x):
                if isinstance(x, tensor_array_ops.TensorArray):
                    return x
                return array_ops.identity(x)
            body_result = nest.map_structure(map_fn, body_result, expand_composites=True)
    body_result = variable_utils.convert_variables_to_tensors(body_result)
    nest.assert_same_structure(list(packed_vars_for_body), list(body_result), expand_composites=True)
    original_body_result = body_result
    result = nest.map_structure(_convert_tensorarray_to_flow, nest.flatten(body_result, expand_composites=True), expand_composites=True)
    result = ops.convert_n_to_tensor_or_composite(result)
    if len(merge_vars) != len(result):
        raise ValueError(f"Number of inputs and outputs of 'body' must match 'loop_vars'. Got {len(merge_vars)} for the number of inputs/outputs, and {len(result)} for 'loop_vars'.")
    next_vars = []
    for m, v in zip(merge_vars, result):
        next_vars.append(_AddNextAndBackEdge(m, v))
    exit_vars = [exit(x[0]) for x in switch_vars]
    self._loop_exits = exit_vars
    self.ExitResult(exit_vars)
    return (original_body_result, exit_vars)