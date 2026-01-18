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
class _PforInput:
    """Input object passed to registered pfor converters."""
    __slots__ = ['pfor', '_op', '_inputs']

    def __init__(self, pfor, op, inputs):
        """Creates a _PforInput object.

    Args:
      pfor: PFor converter object.
      op: the Operation object that is being converted.
      inputs: list of WrappedTensor objects representing converted values of the
        inputs of `op`.
    """
        self.pfor = pfor
        self._op = op
        self._inputs = inputs

    def stack_inputs(self, stack_indices=None, tile_variants=False):
        """Stacks unstacked inputs at `stack_indices`.

    Args:
      stack_indices: indices of inputs at which stacking is done. If None,
        stacking is done at all indices.
      tile_variants: If True, affected indices which have a variant dtype will
        be tiled after this operation to match the expected shape of a
        vectorized tensor. Variants generally need to be un-tiled when they are
        inputs to operations and tiled when returned.
    """
        if stack_indices is None:
            stack_indices = range(len(self._inputs))
        length = self.pfor.loop_len_vector
        for i in stack_indices:
            inp = self._inputs[i]
            is_variant = inp.t.dtype == dtypes.variant
            if not inp.is_stacked:
                self._inputs[i] = _stack(inp.t, length)
                if tile_variants and is_variant:
                    self._inputs[i] = wrap(_tile_variant_with_length(self._inputs[i].t, length), True)
            elif not tile_variants and is_variant:
                self._inputs[i] = wrap(_untile_variant(self._inputs[i].t), True)

    def expanddim_inputs_for_broadcast(self):
        """Reshapes stacked inputs to prepare them for broadcast.

    Since stacked inputs have an extra leading dimension, automatic broadcasting
    rules could incorrectly try to expand dimensions before that leading
    dimension. To avoid that, we reshape these stacked inputs to the maximum
    rank they will need to be broadcasted to.
    """
        if not self._inputs:
            return

        def _get_rank(x):
            rank = array_ops.rank(x.t)
            if not x.is_stacked:
                rank += 1
            return rank
        ranks = [_get_rank(x) for x in self._inputs]
        max_rank = ranks[0]
        for rank in ranks[1:]:
            max_rank = math_ops.maximum(rank, max_rank)
        for i, inp in enumerate(self._inputs):
            if inp.is_stacked:
                shape = array_ops.shape(inp.t)
                rank_diff = array_ops.reshape(max_rank - ranks[i], [1])
                ones = constant_op.constant([1], dtype=shape.dtype)
                ones = array_ops.tile(ones, rank_diff)
                new_shape = array_ops.concat([shape[:1], ones, shape[1:]], axis=0)
                self._inputs[i] = wrap(array_ops.reshape(inp.t, new_shape), True)

    @property
    def inputs(self):
        return self._inputs

    @property
    def num_inputs(self):
        return len(self._inputs)

    def input(self, index):
        assert len(self._inputs) > index, (index, self._inputs)
        return self._inputs[index]

    def stacked_input(self, index):
        t, is_stacked, _ = self.input(index)
        if not is_stacked:
            op_type = self.op_type
            op_def = getattr(self._op, 'op_def', None)
            if op_def is None:
                input_name = 'at index %d' % index
            else:
                input_name = '"%s"' % op_def.input_arg[index].name
            raise ConversionNotImplementedError(f"Input {input_name} of op '{op_type}' expected to be not loop invariant.")
        return t

    def unstacked_input(self, index):
        t, is_stacked, _ = self.input(index)
        if is_stacked:
            op_type = self.op_type
            op_def = getattr(self._op, 'op_def', None)
            if op_def is None:
                input_name = 'at index %d' % index
            else:
                input_name = '"%s"' % op_def.input_arg[index].name
            raise ConversionNotImplementedError(f"Input {input_name} of op '{op_type}' expected to be loop invariant.")
        return t

    @property
    def op(self):
        return self._op

    @property
    def op_type(self):
        return self._op.type

    def get_attr(self, attr):
        return self._op.get_attr(attr)

    @property
    def outputs(self):
        return self._op.outputs

    def output(self, index):
        assert index < len(self._op.outputs)
        return self._op.outputs[index]