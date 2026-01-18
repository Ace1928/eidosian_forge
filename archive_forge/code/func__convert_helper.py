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
def _convert_helper(self, op_or_tensor):
    stack = collections.deque([op_or_tensor])
    while stack:
        y = stack[0]
        if y in self._conversion_map:
            assert isinstance(self._conversion_map[y], (WrappedTensor, ops.Operation))
            stack.popleft()
            continue
        if isinstance(y, ops.Operation):
            assert not y.outputs, ('We only support converting Operation objects with no outputs. Got %s', y)
            y_op = y
        else:
            assert isinstance(y, tensor_lib.Tensor), y
            y_op = y.op
        is_while_loop = y_op.type == 'Exit'
        if is_while_loop:
            while_op = WhileOp(y, pfor_ops=self._pfor_ops, fallback_to_while_loop=self.fallback_to_while_loop, pfor_config=self._pfor_config)
            is_inside_loop = while_op.is_inside_loop
            if is_inside_loop:
                y_op = while_op
        else:
            is_inside_loop = self.op_is_inside_loop(y_op)

        def _add_to_stack(x):
            if x not in self._conversion_map:
                stack.appendleft(x)
                return True
            else:
                return False
        if is_inside_loop:
            added_to_stack = False
            for inp in y_op.inputs:
                added_to_stack |= _add_to_stack(inp)
            for cinp in y_op.control_inputs:
                if cinp.outputs:
                    for t in cinp.outputs:
                        added_to_stack |= _add_to_stack(t)
                else:
                    added_to_stack |= _add_to_stack(cinp)
            if added_to_stack:
                continue
            converted_inputs = [self._conversion_map[inp] for inp in y_op.inputs]
            some_input_converted = any((self._was_converted(x) for x in y_op.inputs))
            some_input_stacked = any((x.is_stacked for x in converted_inputs))
            converted_control_ops = set()
            some_control_input_converted = False
            for cinp in y_op.control_inputs:
                if cinp.outputs:
                    for t in cinp.outputs:
                        converted_t = self._conversion_map[t]
                        if self._was_converted(t):
                            some_control_input_converted = True
                        converted_control_ops.add(converted_t.t.op)
                else:
                    converted_cinp = self._conversion_map[cinp]
                    assert isinstance(converted_cinp, ops.Operation)
                    if converted_cinp != cinp:
                        some_control_input_converted = True
                    converted_control_ops.add(converted_cinp)
            converted_control_ops = list(converted_control_ops)
            is_stateful = _is_stateful_pfor_op(y_op)
        else:
            converted_inputs = []
            converted_control_ops = []
        logging.vlog(3, 'converting op:%s\ninputs:%s\ncontrol_inputs:%s', y_op, converted_inputs, converted_control_ops)
        control_dependencies = [] if is_while_loop else converted_control_ops
        with ops.control_dependencies(control_dependencies), ops.name_scope(y_op.name + '/pfor/'), ops.get_default_graph()._original_op(y_op):
            reduce_output = self._convert_reduction(y)
            if reduce_output is not None:
                new_outputs = reduce_output
            elif (not is_inside_loop or (not is_stateful and (not some_input_converted) and (not some_control_input_converted))) and y.graph == ops.get_default_graph():
                if y is y_op:
                    assert not isinstance(y_op, WhileOp)
                    new_outputs = y_op
                else:
                    new_outputs = [wrap(x, False) for x in y_op.outputs]
            elif not (is_stateful or is_while_loop or some_input_stacked):
                new_op = _create_op(y_op.type, [x.t for x in converted_inputs], [x.dtype for x in y_op.outputs], y_op.node_def.attr)
                if y is y_op:
                    new_outputs = new_op
                else:
                    new_outputs = []
                    for old_output, new_output in zip(y_op.outputs, new_op.outputs):
                        handle_data_util.copy_handle_data(old_output, new_output)
                        new_outputs.append(wrap(new_output, False))
            else:
                if hasattr(y_op, 'pfor_converter'):
                    converter = y_op.pfor_converter
                else:
                    converter = _pfor_converter_registry.get(y_op.type, None)
                if converter is None:
                    root_cause = f'there is no registered converter for this op.'
                    has_variant_outputs = any((x.dtype == dtypes.variant for x in y_op.outputs))
                    has_vectorized_variant_inputs = any((_is_variant_with_internal_stacking(x) for x in y_op.inputs))
                    if self._fallback_to_while_loop and (not has_variant_outputs) and (not has_vectorized_variant_inputs):
                        converter = partial(_fallback_converter, root_cause=root_cause, warn=self._warn)
                    else:
                        message = f'No pfor vectorization defined for {y_op.type}\n{y_op}\n inputs: {converted_inputs}.'
                        if not self._fallback_to_while_loop:
                            message += 'Consider enabling the fallback_to_while_loop option to pfor, which may run slower.'
                        raise ValueError(message)
                pfor_inputs = _PforInput(self, y_op, converted_inputs)
                try:
                    try:
                        new_outputs = converter(pfor_inputs)
                    except ConversionNotImplementedError as e:
                        has_vectorized_variant_inputs = any((_is_variant_with_internal_stacking(x) for x in y_op.inputs))
                        if self._fallback_to_while_loop and (not has_vectorized_variant_inputs):
                            new_outputs = _fallback_converter(pfor_inputs, root_cause=str(e))
                        else:
                            raise ValueError(str(e)).with_traceback(sys.exc_info()[2])
                except Exception as e:
                    logging.error(f'Got error while pfor was converting op {y_op} with inputs {y_op.inputs[:]}\n, converted inputs {pfor_inputs.inputs}\nHere are the pfor conversion stack traces: {e}')
                    original_op = y_op
                    while isinstance(original_op, ops.Operation):
                        logging.error('%s\ncreated at:\n  %s', original_op, '  '.join(traceback.format_list(original_op.traceback)))
                        original_op = original_op._original_op
                    raise
                if isinstance(new_outputs, WrappedTensor):
                    new_outputs = [new_outputs]
                assert isinstance(new_outputs, (list, tuple, ops.Operation)), new_outputs
            logging.vlog(2, f'converted {y_op} {new_outputs}')
            if y is y_op:
                assert isinstance(new_outputs, ops.Operation)
                self._add_conversion(y_op, new_outputs)
            else:
                assert len(y_op.outputs) == len(new_outputs), (y_op, y_op.outputs, new_outputs)
                for old_output, new_output in zip(y_op.outputs, new_outputs):
                    assert isinstance(new_output, WrappedTensor), (new_output, y, y_op)
                    assert old_output.dtype == new_output.t.dtype, (new_output, y, y_op)
                    output_shape = old_output.shape
                    if not new_output.is_sparse_stacked:
                        if new_output.is_stacked:
                            loop_len = tensor_util.constant_value(self.loop_len_vector)
                            if loop_len is None:
                                batch_dim = tensor_shape.TensorShape([None])
                            else:
                                batch_dim = tensor_shape.TensorShape(loop_len)
                            output_shape = batch_dim.concatenate(output_shape)
                        if _is_variant_with_internal_stacking(new_output.t):
                            new_output.t.set_shape([])
                        else:
                            new_output.t.set_shape(output_shape)
                    self._add_conversion(old_output, new_output)
            stack.popleft()
    return self._conversion_map[op_or_tensor]