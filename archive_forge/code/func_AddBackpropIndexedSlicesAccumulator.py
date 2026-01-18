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
def AddBackpropIndexedSlicesAccumulator(self, op, grad):
    """This is used for accumulating gradients that are IndexedSlices.

    This is essentially the equivalent of AddBackpropAccumulator but optimized
    for things like updating embeddings from within a while loop.

    Args:
      op: The Enter op for a loop invariant.
      grad: The partial gradients represented as an IndexedSlices.

    Returns:
      The accumulated IndexedSlices gradient of the loop invariant.
    """
    values = grad.values
    indices = grad.indices
    dense_shape = grad.dense_shape
    self.Exit()
    if self.outer_context:
        self.outer_context.Enter()
    if values.get_shape().is_fully_defined():
        values_shape = tensor_shape.TensorShape([tensor_shape.Dimension(1)] + values.get_shape().dims[1:])
        if self.outer_context:
            self.outer_context.Enter()
        values_acc = constant_op.constant(0, values.dtype, shape=values_shape, name='b_acc')
        if self.outer_context:
            self.outer_context.Exit()
    else:
        values_shape = _resource_safe_shape(op.inputs[0])[1:]
        values_shape = array_ops.concat([[1], values_shape], 0)
        values_acc = array_ops.zeros(values_shape, dtype=values.dtype)
    indices_acc = constant_op.constant([0], indices.dtype)
    shape_acc = None
    if dense_shape is not None:
        if dense_shape.get_shape().is_fully_defined():
            if self.outer_context:
                self.outer_context.Enter()
            shape_acc = constant_op.constant(0, dense_shape.dtype, shape=dense_shape.get_shape())
            if self.outer_context:
                self.outer_context.Exit()
        else:
            shape_acc = array_ops.zeros_like(array_ops.shape_internal(op.inputs[0], optimize=False, out_type=dense_shape.dtype), optimize=False)
    if self.outer_context:
        self.outer_context.Exit()
    self.Enter()
    self.AddName(values_acc.name)
    self.AddName(indices_acc.name)
    init_acc = [indices_acc, values_acc]
    if shape_acc is not None:
        self.AddName(shape_acc.name)
        init_acc.append(shape_acc)
    enter_acc = [_Enter(x, self._name, is_constant=False, parallel_iterations=self._parallel_iterations, use_input_shape=False, name='b_acc') for x in init_acc]
    enter_acc[0].set_shape([None])
    if values_acc.shape.dims is not None:
        enter_acc[1].set_shape([None] + values_acc.shape.as_list()[1:])
    self.loop_enters.extend(enter_acc)
    merge_acc = [merge([x, x], name='b_acc')[0] for x in enter_acc]
    switch_acc = [switch(x, self._pivot) for x in merge_acc]
    acc_indexed_slices = [array_ops.concat([xa[1], xv], 0) for xa, xv in zip(switch_acc[:2], [indices, values])]
    if shape_acc is not None:
        acc_indexed_slices.append(math_ops.maximum(dense_shape, switch_acc[2][1]))
    next_acc = [_NextIteration(x) for x in acc_indexed_slices]
    for xm, xn in zip(merge_acc, next_acc):
        xm.op._update_input(1, xn)
    exit_acc = [exit(x[0], name='b_acc') for x in switch_acc]
    self.loop_exits.extend(exit_acc)
    self.ExitResult(exit_acc)
    return indexed_slices.IndexedSlices(indices=exit_acc[0], values=exit_acc[1], dense_shape=exit_acc[2] if shape_acc is not None else None)