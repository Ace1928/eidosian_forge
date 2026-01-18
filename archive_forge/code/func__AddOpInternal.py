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
def _AddOpInternal(self, op):
    """Add `op` to the current context.

    We move any external control dependencies of the op to the loop pivot, to
    ensure they get executed.
    """
    if op.type in ['PartitionedCall', 'StatefulPartitionedCall']:
        op._add_control_input(self.GetControlPivot().op)
    if not op.inputs:
        control_inputs, external_inputs = self._RemoveExternalControlEdges(op)
        if not control_inputs:
            op._add_control_input(self.GetControlPivot().op)
        for x in op.outputs:
            self._values.add(x.name)
    else:
        for index in range(len(op.inputs)):
            x = op.inputs[index]
            real_x = self.AddValue(x)
            if real_x != x:
                op._update_input(index, real_x)
        _, external_inputs = self._RemoveExternalControlEdges(op)
        self._MaybeAddControlDependency(op)
        for x in op.outputs:
            self._values.add(x.name)
    if external_inputs:
        with ops.control_dependencies(None):
            self.Enter()
            external_inputs = [array_ops.identity(x.outputs[0]).op for x in external_inputs if x.outputs]
            self.Exit()
        op._add_control_inputs(external_inputs)
    if self._outer_context or not util.IsLoopExit(op):
        op.graph.prevent_fetching(op)
        for x in op.outputs:
            op.graph.prevent_feeding(x)
    if self._outer_context:
        self._outer_context.AddInnerOp(op)