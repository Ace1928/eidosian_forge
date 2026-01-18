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
class PFor:
    """Implementation of rewrite of parallel-for loops.

  This class takes a DAG or a set of DAGs representing the body of a
  parallel-for loop, and adds new operations to the graph that implements
  functionality equivalent to running that loop body for a specified number of
  iterations. This new set of nodes may or may not use a tensorflow loop
  construct.

  The process of conversion does not delete or change any existing operations.
  It only adds operations that efficiently implement the equivalent
  functionality. We refer to the added ops as "converted ops".

  The conversion process uses a simple greedy heuristic. It walks the loop body
  and tries to express the functionality of running each node in a loop with a
  new set of nodes. When converting an op several cases are possible:
  - The op is not inside the loop body. Hence it can be used as is.
  - The op does not depend on the iteration number and is stateless. In this
    case, it can be used as is.
  - The op is not stateful, and depends on iteration number only through control
    dependencies. In this case, we can create a single op with same inputs and
    attributes, but with "converted" control dependencies.
  - The op is not stateful, and all its inputs are loop invariant. In this
    case, similar to above, we can create a single op with same inputs and
    attributes, but with "converted" control dependencies.
  - The op is stateful or at least one of the inputs is not loop invariant. In
    this case, we run the registered converter for that op to create a set of
    converted ops. All nodes in the set will have converted control dependencies
    corresponding to control dependencies of the original op. If the op returned
    multiple outputs, "converted outputs" could be produced by different ops in
    this set.
  """

    def __init__(self, loop_var, loop_len, pfor_ops, fallback_to_while_loop, all_indices=None, all_indices_partitioned=False, pfor_config=None, warn=False):
        """Creates an object to rewrite a parallel-for loop.

    Args:
      loop_var: Tensor output of a Placeholder operation. The value should
        be an int32 scalar representing the loop iteration number.
      loop_len: A scalar or scalar Tensor representing the number of iterations
        the loop is run for.
      pfor_ops: List of all ops inside the loop body.
      fallback_to_while_loop: If True, on failure to vectorize an op, a while
        loop is used to sequentially execute that op.
      all_indices: If not None, an int32 vector with size `loop_len`
        representing the iteration ids that are still active. These values
        should be unique and sorted. However they may not be contiguous. This is
        typically the case when inside a control flow construct which has
        partitioned the indices of the iterations that are being converted.
      all_indices_partitioned: If True, this object is being constructed from a
        control flow construct where not all the pfor iterations are guaranteed
        to be active.
      pfor_config: PForConfig object used while constructing the loop body.
      warn: Whether or not to warn on while loop conversions.
    """
        assert isinstance(loop_var, tensor_lib.Tensor)
        assert loop_var.op.type == 'PlaceholderWithDefault'
        self._loop_var = loop_var
        loop_len_value = tensor_util.constant_value(loop_len)
        if loop_len_value is not None:
            loop_len = loop_len_value
            self._loop_len_vector = ops.convert_to_tensor([loop_len])
        else:
            self._loop_len_vector = array_ops.reshape(loop_len, [1])
        self._all_indices_partitioned = all_indices_partitioned
        if all_indices_partitioned:
            assert all_indices is not None
        self.all_indices = math_ops.range(loop_len) if all_indices is None else all_indices
        self._conversion_map = object_identity.ObjectIdentityDictionary()
        self._conversion_map[loop_var] = wrap(self.all_indices, True)
        self._pfor_ops = set(pfor_ops)
        self._pfor_op_ids = set((x._id for x in pfor_ops))
        self._fallback_to_while_loop = fallback_to_while_loop
        self._warn = warn
        self._pfor_config = pfor_config

    def op_is_inside_loop(self, op):
        """True if op was created inside the pfor loop body."""
        assert isinstance(op, ops.Operation)
        return op._id in self._pfor_op_ids

    def _convert_sparse(self, y):
        """Returns the converted value corresponding to SparseTensor y.

    For SparseTensors, instead of stacking the component tensors separately,
    resulting in component tensors with shapes (N, m, rank), (N, m), and (N,
    rank) respectively for indices, values, and dense_shape (where N is the loop
    length and m is the number of sparse tensor values per loop iter), we want
    to logically stack the SparseTensors, to create a SparseTensor whose
    components are size (N * m, rank + 1), (N * m, ), and (rank + 1,)
    respectively.

    Here, we try to get the conversion of each component tensor.
    If the tensors are stacked via a sparse conversion, return the resulting
    SparseTensor composed of the converted components. Otherwise, the component
    tensors are either unstacked or stacked naively. In the latter case, we
    unstack the component tensors to reform loop_len SparseTensor elements,
    then correctly batch them.

    The unstacked tensors must have the same rank. Each dimension of each
    SparseTensor will expand to be the largest among all SparseTensor elements
    for that dimension. For example, if there are N SparseTensors of rank 3
    being stacked, with N dense shapes, where the i_th shape is (x_i, y_i, z_i),
    the new dense shape will be (N, max_i(x_i), max_i(y_i), max_i(z_i)).

    Args:
      y: A tf.sparse.SparseTensor.

    Returns:
      A tf.sparse.SparseTensor that is the converted value corresponding to y.
    """
        outputs = [self._convert_helper(t) for t in (y.indices, y.values, y.dense_shape)]
        assert all((isinstance(o, WrappedTensor) for o in outputs))
        if all((w.is_sparse_stacked for w in outputs)):
            return sparse_tensor.SparseTensor(*[w.t for w in outputs])
        assert not any((w.is_sparse_stacked for w in outputs)), 'Error converting SparseTensor. All components should be logically stacked, or none.'
        return self._restack_sparse_tensor_logically(*[self._unwrap_or_tile(w) for w in outputs])

    def _restack_sparse_tensor_logically(self, indices, values, shape):
        sparse_tensor_rank = indices.get_shape().dims[-1].value
        if sparse_tensor_rank is not None:
            sparse_tensor_rank += 1

        def fn(args):
            res = gen_sparse_ops.serialize_sparse(args[0], args[1], args[2], out_type=dtypes.variant)
            return res
        result = map_fn.map_fn(fn, [indices, values, shape], dtype=dtypes.variant)
        return sparse_ops.deserialize_sparse(result, dtype=values.dtype, rank=sparse_tensor_rank)

    def _unwrap_or_tile(self, wrapped_tensor):
        """Given a wrapped tensor, unwrap if stacked. Otherwise, tiles it."""
        output, is_stacked = (wrapped_tensor.t, wrapped_tensor.is_stacked)
        if is_stacked:
            return output
        else:
            return _stack(output, self._loop_len_vector).t

    def convert(self, y):
        """Returns the converted value corresponding to y.

    Args:
      y: A Tensor or a ops.Operation object. If latter, y should not have
        any outputs.

    Returns:
      If y does not need to be converted, it returns y as is. Else it returns
      the "converted value" corresponding to y.
    """
        if y is None:
            return None
        if isinstance(y, sparse_tensor.SparseTensor):
            return self._convert_sparse(y)
        assert isinstance(y, (tensor_lib.Tensor, ops.Operation)), y
        output = self._convert_helper(y)
        if isinstance(output, WrappedTensor):
            assert isinstance(y, tensor_lib.Tensor)
            return self._unwrap_or_tile(output)
        else:
            assert isinstance(y, ops.Operation)
            assert not y.outputs
            assert isinstance(output, ops.Operation)
        return output

    def _was_converted(self, t):
        """True if t is not a conversion of itself."""
        converted_t = self._conversion_map[t]
        return converted_t.t is not t

    def _add_conversion(self, old_output, new_output):
        assert isinstance(old_output, (tensor_lib.Tensor, ops.Operation)), old_output
        assert isinstance(new_output, (WrappedTensor, ops.Operation)), new_output
        self._conversion_map[old_output] = new_output

    def _convert_reduction(self, y):
        if self._pfor_config is None or isinstance(y, ops.Operation):
            return None
        reduction = self._pfor_config._lookup_reduction(y)
        if reduction is None:
            return None
        reduction_fn, reduction_args = reduction
        batched_args = []
        for reduction_arg in reduction_args:
            assert isinstance(reduction_arg, tensor_lib.Tensor), reduction_arg
            assert reduction_arg in self._conversion_map, 'Unable to handle reduction of %s, possibly as it was used inside a control flow construct. Note that reductions across pfor iterations are currently not supported inside control flow constructs.' % reduction_arg
            batched_arg = self._conversion_map[reduction_arg]
            batched_args.append(self._unwrap_or_tile(batched_arg))
        outputs = reduction_fn(*batched_args)
        return [wrap(output, False) for output in nest.flatten(outputs)]

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

    @property
    def loop_len_vector(self):
        """Returns a single element vector whose value is number of iterations."""
        return self._loop_len_vector

    @property
    def loop_var(self):
        """Returns placeholder loop variable."""
        return self._loop_var

    @property
    def pfor_ops(self):
        return self._pfor_ops

    @property
    def pfor_config(self):
        return self._pfor_config

    @property
    def all_indices_partitioned(self):
        """all_indices_partitioned property.

    Returns:
      True if we are inside a control flow construct and not all pfor iterations
      may be active.
    """
        return self._all_indices_partitioned

    @property
    def fallback_to_while_loop(self):
        return self._fallback_to_while_loop