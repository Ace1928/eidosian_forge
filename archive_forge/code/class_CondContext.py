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
class CondContext(ControlFlowContext):
    """The context for the conditional construct."""

    def __init__(self, pred=None, pivot=None, branch=None, name='cond_text', context_def=None, import_scope=None):
        """Creates a `CondContext`.

    Args:
      pred: The `boolean` tensor for the conditional predicate.
      pivot: The predicate tensor in this branch.
      branch: 0 or 1 representing this branch.
      name: Name of the `CondContext` python object.
      context_def: Optional `ContextDef` protocol buffer to initialize the
        `CondContext` object from.
      import_scope: Optional `string`. Name scope to add. Only used when
        initialing from protocol buffer.
    """
        self._name = ops.get_default_graph().unique_name(name)
        if context_def:
            self._init_from_proto(context_def, import_scope=import_scope)
        else:
            ControlFlowContext.__init__(self)
            self._pred = pred
            self._pivot = pivot
            self._branch = branch
            self._values.add(pred.name)
            self._external_values[pred.name] = pred
            self._values.add(pivot.name)
            pivot.op._set_control_flow_context(self)

    def _init_from_proto(self, context_def, import_scope=None):
        """Creates a new `CondContext` from protocol buffer.

    Args:
      context_def: `CondContextDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
        assert isinstance(context_def, control_flow_pb2.CondContextDef)
        g = ops.get_default_graph()
        self._name = ops.prepend_name_scope(context_def.context_name, import_scope)
        self._pred = g.as_graph_element(ops.prepend_name_scope(context_def.pred_name, import_scope))
        self._pivot = g.as_graph_element(ops.prepend_name_scope(context_def.pivot_name, import_scope))
        self._branch = context_def.branch
        super(CondContext, self).__init__(values_def=context_def.values_def, import_scope=import_scope)

    @property
    def pred(self):
        return self._pred

    @property
    def pivot(self):
        return self._pivot

    @property
    def branch(self):
        return self._branch

    @property
    def grad_state(self):
        if self.GetWhileContext():
            return self.GetWhileContext().grad_state
        return None

    @property
    def back_prop(self):
        if self.GetWhileContext():
            return self.GetWhileContext().back_prop
        return False

    def GetControlPivot(self):
        return self._pivot

    def to_proto(self, export_scope=None):
        """Converts a `CondContext` to a `CondContextDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `CondContextDef` protocol buffer.
    """
        if export_scope is None or self.name.startswith(export_scope):
            context_def = control_flow_pb2.CondContextDef()
            context_def.context_name = ops.strip_name_scope(self.name, export_scope)
            context_def.pred_name = ops.strip_name_scope(self._pred.name, export_scope)
            context_def.pivot_name = ops.strip_name_scope(self._pivot.name, export_scope)
            context_def.branch = self._branch
            context_def.values_def.MergeFrom(super(CondContext, self)._to_values_def(export_scope))
            for nested in self._nested_contexts:
                nested_def = context_def.nested_contexts.add()
                nested.to_control_flow_context_def(nested_def)
            return context_def
        else:
            return None

    @staticmethod
    def from_proto(context_def, import_scope=None):
        """Returns a `CondContext` object created from `context_def`."""
        ret = CondContext(context_def=context_def, import_scope=import_scope)
        ret.Enter()
        for nested_def in context_def.nested_contexts:
            from_control_flow_context_def(nested_def, import_scope=import_scope)
        ret.Exit()
        return ret

    def to_control_flow_context_def(self, context_def, export_scope=None):
        context_def.cond_ctxt.CopyFrom(self.to_proto(export_scope=export_scope))

    def AddValue(self, val):
        """Add `val` to the current context and its outer context recursively."""
        if val.name in self._values:
            result = self._external_values.get(val.name)
            result = val if result is None else result
        else:
            result = val
            self._values.add(val.name)
            if self._outer_context:
                result = self._outer_context.AddValue(val)
                self._values.add(result.name)
                self._external_values[result.name] = result
            with ops.control_dependencies(None):
                result = _SwitchRefOrTensor(result, self._pred)[self._branch]
                if self._outer_context:
                    self._outer_context.AddInnerOp(result.op)
            result.op.graph.prevent_fetching(result.op)
            result.op._set_control_flow_context(self)
            ctxt = self
            while ctxt is not None:
                ctxt._values.add(result.name)
                ctxt = ctxt._outer_context
            self._external_values[val.name] = result
        return result

    def AddOp(self, op):
        self._AddOpInternal(op)

    def _AddOpInternal(self, op):
        """Add `op` to the current context."""
        if not op.inputs:
            self._RemoveExternalControlEdges(op)
            if not any((util.OpInContext(input_op, self) for input_op in op.control_inputs)):
                op._add_control_input(self._pivot.op)
        else:
            for index in range(len(op.inputs)):
                x = op.inputs[index]
                if op.type == 'Merge' and x.op.type == 'NextIteration':
                    real_x = x
                else:
                    real_x = self.AddValue(x)
                if real_x != x:
                    op._update_input(index, real_x)
            self._RemoveExternalControlEdges(op)
            if op.graph._is_function(op.type) or op.type == 'SymbolicGradient':
                op._add_control_input(self._pivot.op)
        output_names = [x.name for x in op.outputs]
        ctxt = self
        while ctxt is not None:
            ctxt._values.update(output_names)
            ctxt = ctxt._outer_context
        if self._outer_context or not util.IsLoopExit(op):
            op.graph.prevent_fetching(op)
        if self._outer_context:
            self._outer_context.AddInnerOp(op)

    def _ProcessOutputTensor(self, val):
        """Process an output tensor of a conditional branch."""
        real_val = val
        if val.name not in self._values:
            self._values.add(val.name)
            if self._outer_context:
                real_val = self._outer_context.AddValue(val)
                self._values.add(real_val.name)
                self._external_values[real_val.name] = real_val
            real_val = _SwitchRefOrTensor(real_val, self._pred)[self._branch]
            self._external_values[val.name] = real_val
        else:
            external_val = self._external_values.get(val.name)
            if external_val is not None:
                real_val = external_val
        return real_val

    def _BuildCondTensor(self, v):
        if isinstance(v, ops.Operation):
            return with_dependencies([v], self._pivot)
        else:
            v = nest.map_structure(_convert_tensorarray_to_flow, v, expand_composites=True)
            return self._ProcessOutputTensor(ops.convert_to_tensor(v))

    def BuildCondBranch(self, fn):
        """Add the subgraph defined by fn() to the graph."""
        pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)
        original_result = fn()
        post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)
        if len(post_summaries) > len(pre_summaries):
            new_summaries = post_summaries[len(pre_summaries):]
            summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)
            summary_ref[:] = pre_summaries
            with ops.control_dependencies(new_summaries):
                if original_result is None:
                    return (no_op(), None)
                elif not isinstance(original_result, ops.Operation):
                    original_result = variable_utils.convert_variables_to_tensors(original_result)
                    original_result = nest.map_structure(array_ops.identity, original_result, expand_composites=True)
        if original_result is None:
            return (None, None)
        original_result = variable_utils.convert_variables_to_tensors(original_result)
        result = nest.map_structure(self._BuildCondTensor, original_result, expand_composites=True)
        if not isinstance(result, (list, _basetuple)):
            result = [result]
        return (original_result, result)

    def IsCondContext(self):
        return True