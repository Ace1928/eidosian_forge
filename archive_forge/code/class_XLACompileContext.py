import contextlib
from tensorflow.compiler.jit.ops import xla_ops
from tensorflow.compiler.jit.ops import xla_ops_grad  # pylint: disable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import summary_op_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
class XLACompileContext(control_flow_ops.XLAControlFlowContext):
    """A `ControlFlowContext` for nodes inside an XLA computation cluster.

  THIS IS ONLY FOR TENSORFLOW INTERNAL IMPLEMENTATION, DO NO USE DIRECTLY.

  The primary role of `XLACompileContext` is to mark operators inside a
  xla.compile() computation with attribute "_xla_compile_id=XYZ", where XYZ is
  a unique name.

  `ControlFlowContext` is used to perform the annotation since it integrates
  with Tensorflow constructs like ResourceVariables. For example, if a
  `ResourceVariable` is constructed inside a xla.compile() block, the
  `ResourceVariable` implementation can use
  `with ops.control_dependencies(None)` to build the variable's definition
  outside the compiled computation.
  """

    def __init__(self, name, pivot):
        """Builds a new XLACompileContext.

    Args:
      name: a unique name for the context, used to populate the
        `_xla_compile_id` attribute.
      pivot: a pivot node. Nodes in the XLACompileContext that do not have any
        inputs will have a control dependency on the pivot node. This ensures
        that nodes are correctly included in any enclosing control flow
        contexts.
    """
        super(XLACompileContext, self).__init__()
        self._name = name
        self._name_as_bytes = compat.as_bytes(name)
        self._unsupported_ops = []
        self._pivot = pivot

    def report_unsupported_operations(self):
        if self._unsupported_ops:
            op_str = '\n'.join(['  %s (%s)' % (op.type, op.name) for op in self._unsupported_ops[:_MAX_WARNING_LINES]])
            logging.warning('%d unsupported operations found: \n%s', len(self._unsupported_ops), op_str)
            if len(self._unsupported_ops) > _MAX_WARNING_LINES:
                logging.warning('... and %d more', len(self._unsupported_ops) - _MAX_WARNING_LINES)

    def _RemoveExternalControlEdges(self, op):
        """Remove any external control dependency on this op."""
        internal_control_inputs = []
        external_control_inputs = []
        for x in op.control_inputs:
            is_internal_op = False
            ctxt = x._get_control_flow_context()
            while ctxt is not None:
                if ctxt == self:
                    is_internal_op = True
                    break
                ctxt = ctxt._outer_context
            if is_internal_op:
                internal_control_inputs.append(x)
            else:
                external_control_inputs.append(x)
        op._remove_all_control_inputs()
        op._add_control_inputs(internal_control_inputs)
        return (internal_control_inputs, external_control_inputs)

    def AddOp(self, op):
        """Create op in XLACompileContext and notifies outer context recursively."""
        if op.type in _DENYLISTED_OPS:
            logging.error('Operation of type %s (%s) is not supported in XLA. Execution will fail if this op is used in the graph. ', op.type, op.name)
        if op.type in _UNSUPPORTED_OPS:
            self._unsupported_ops.append(op)
        if any((x.dtype._is_ref_dtype for x in op.inputs)):
            raise NotImplementedError('Non-resource Variables are not supported inside XLA computations (operator name: %s)' % op.name)
        if _XLA_COMPILE_ATTR in op.node_def.attr:
            raise ValueError('XLA compiled computations cannot be nested, (operator name: %s)' % op.name)
        op._set_attr(_XLA_COMPILE_ATTR, attr_value_pb2.AttrValue(s=self._name_as_bytes))
        op.graph.prevent_feeding(op)
        op.graph.prevent_fetching(op)
        internal_control_inputs, external_control_inputs = self._RemoveExternalControlEdges(op)
        if not op.inputs:
            if not internal_control_inputs:
                op._add_control_input(self._pivot)
        else:
            for index in range(len(op.inputs)):
                x = op.inputs[index]
                real_x = self.AddValue(x)
                if real_x is not x:
                    op._update_input(index, real_x)
        if external_control_inputs:
            with ops.control_dependencies(None):
                self.Enter()
                external_control_inputs = [array_ops.identity(x.outputs[0]).op for x in external_control_inputs if x.outputs]
                self.Exit()
            op._add_control_inputs(external_control_inputs)
        output_names = [x.name for x in op.outputs]
        context = self
        while context is not None:
            context._values.update(output_names)
            context = context._outer_context
        if self._outer_context:
            self._outer_context.AddInnerOp(op)

    def AddValue(self, val):
        """Add `val` to the current context and its outer context recursively."""
        if val.name in self._values:
            result = self._external_values.get(val.name)
            return val if result is None else result
        result = val
        self._values.add(val.name)
        if self._outer_context:
            result = self._outer_context.AddValue(val)
            self._values.add(result.name)
        self._external_values[val.name] = result
        return result

    def AddInnerOp(self, op):
        self.AddOp(op)
        if self._outer_context:
            self._outer_context.AddInnerOp(op)

    @property
    def grad_state(self):
        return None

    @property
    def back_prop(self):
        """Forwards to the enclosing while context, if any."""
        if self.GetWhileContext():
            return self.GetWhileContext().back_prop
        return False