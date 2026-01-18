import collections
import enum
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import auto_control_deps_utils as utils
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import registry
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
class AutomaticControlDependencies(object):
    """Context manager to automatically add control dependencies.

  Code under this context manager will act as if a sensible set of control
  dependencies were present. More specifically:
    1. All stateful ops in the scope will execute (with the exception of ops in
       ASYNC_STATEFUL_OPS and LEGACY_RANDOM_OPS)
    2. Stateful ops which modify the same resource will execute in program order

  Note: creating variables in an automatic control dependencies context is not
  supported (the value of the variables will never change as they will keep
  getting reinitialized).

  NOT THREAD SAFE
  """

    def __init__(self):
        self._returned_tensors = object_identity.ObjectIdentitySet()
        self.ops_which_must_run = set()
        self._independent_ops = []

    def mark_as_return(self, tensor):
        """Acts like identity but marks the `Tensor` as a return value.

    This will possibly return a copy of the `Tensor`. Usage:

    ```
      with AutomaticControlDependencies() as a:
       ...
       t = a.mark_as_return(t)
      _ = ...(t...)  # i.e. it's safe to use t here
    ```

    Args:
      tensor: the `Tensor` to be marked

    Returns:
      a copy of the `Tensor`.
    """
        if isinstance(tensor, indexed_slices.IndexedSlices):
            values = array_ops.identity(tensor.values)
            indices = array_ops.identity(tensor.indices)
            self._returned_tensors.add(indices)
            self._returned_tensors.add(values)
            return indexed_slices.IndexedSlices(values, indices, dense_shape=tensor.dense_shape)
        elif isinstance(tensor, sparse_tensor.SparseTensor):
            values = array_ops.identity(tensor.values)
            indices = array_ops.identity(tensor.indices)
            self._returned_tensors.add(indices)
            self._returned_tensors.add(values)
            return sparse_tensor.SparseTensor(indices, values, dense_shape=tensor.dense_shape)
        elif isinstance(tensor, tensor_array_ops.TensorArray):
            flow = array_ops.identity(tensor.flow)
            self._returned_tensors.add(flow)
            return tensor_array_ops.build_ta_with_new_flow(tensor, flow)
        tensor = array_ops.identity(tensor)
        self._returned_tensors.add(tensor)
        return tensor

    def run_independently(self, op):
        """Marks the given op as independent.

    Overrides any other rule for the op.

    Independent ops are guaranteed to execute before the return values, but
    are allowed to run in parallel with everything else. Use in programs which
    can guarantee that an op has side effects that don't affect any other op.

    Args:
      op: An operation
    """
        self._independent_ops.append(op)
        op._set_attr('_independent_side_effects', attr_value_pb2.AttrValue(b=True))

    def __enter__(self):
        if context.executing_eagerly():
            return self
        g = ops.get_default_graph()
        self._graph = g
        g._add_control_dependencies = True
        g.experimental_acd_manager = self
        self._n_operations = g.num_operations()
        return self

    def _process_switch(self, switch_op, ops_which_must_run, last_write_to_resource, merge_for_resource):
        """Processes a switch node for a resource input.

    When tensorflow creates a cond, it creates a control flow context for each
    branch of the cond. Each external tensor accessed by that branch is routed
    through a switch op, which gets created in the graph _after_ the op which
    uses that tensor get created.

    If the resource comes from another switch op we process that one first.

    _process_switch creates a corresponding merge node for the switch node. This
    merge node is added to the outer control flow context of the switch
    node. We also ensure that:

      1. The switch node executes after the previous op which used the resource
         tensor

      2. Any op which uses a resource output of the switch node executes before
         the merge for the switch node.

      3. The next op which uses the input resource to the switch node (which
         might be another switch node for the other branch of the conditional)
         will execute after the merge node is done.

      4. The merge node is marked as must_run so it will run even if no
         subsequent operation uses the resource.

    Args:
      switch_op: the switch op to be processed
      ops_which_must_run: the set of ops which must run
      last_write_to_resource: map from resource tensor to last op updating
        it
      merge_for_resource: map from resource tensor to merge which must follow
        all usages of it.
    """
        inp = switch_op.inputs[0]
        input_id = ops.tensor_id(inp)
        if inp.dtype == dtypes_module.resource and inp.op.type == 'Switch':
            self._process_switch(inp.op, ops_which_must_run, last_write_to_resource, merge_for_resource)
        output = switch_op.outputs[0]
        output_id = ops.tensor_id(output)
        if output_id in merge_for_resource:
            return
        new_merge = control_flow_ops.merge(switch_op.outputs, name='artificial_merge')
        new_merge[0].op._control_flow_context = switch_op._control_flow_context.outer_context
        ops_which_must_run.add(new_merge[0].op)
        if input_id in last_write_to_resource:
            switch_op._add_control_input(last_write_to_resource[input_id])
        last_write_to_resource[input_id] = new_merge[0].op
        if input_id in merge_for_resource:
            merge_for_resource[input_id]._add_control_input(new_merge[0].op)
        for o in switch_op.outputs:
            merge_for_resource[ops.tensor_id(o)] = new_merge[0].op

    def __exit__(self, unused_type, unused_value, unused_traceback):
        if context.executing_eagerly():
            return
        if self._graph is not ops.get_default_graph():
            raise RuntimeError(f'Within the automatic control dependency context, the default graph cannot change. Upon entry it was {self._graph}, but on exit it changed to {ops.get_default_graph()}')
        outer_graph = getattr(self._graph, 'outer_graph', None)
        if outer_graph is not None:
            self._graph._add_control_dependencies = outer_graph._add_control_dependencies
        else:
            self._graph._add_control_dependencies = False
        self._graph.experimental_acd_manager = None
        last_write_to_resource = {}
        reads_since_last_write_to_resource = collections.defaultdict(list)
        collective_manager_scopes_opened = {}
        collective_manager_scopes_used = {}
        ops_which_must_run = set()
        merge_for_resource = {}
        new_operations = self._graph.get_operations()[self._n_operations:]
        for op in new_operations:
            if control_flow_util.IsInWhileLoop(op):
                continue
            control_inputs = set()
            if op.type in MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS:
                self.run_independently(op)
            if op in self._independent_ops:
                ops_which_must_run.add(op)
                continue
            if op_def_registry.get(op.type) is None or (op_is_stateful(op) and (op.type not in utils.RESOURCE_READ_OPS or any((output.consumers() for output in op.outputs)))):
                ops_which_must_run.add(op)
            if op.type == 'NoOp':
                try:
                    collective_manager_scopes_opened[op.get_attr('_collective_manager_id')] = op
                except ValueError:
                    pass
            if op.type == 'Switch' and op.inputs[0].dtype == dtypes_module.resource:
                continue
            if op.type == 'Merge':
                for o in ops_which_must_run:
                    op._add_control_input(o)
                    for inp in o.inputs:
                        input_id = ops.tensor_id(inp)
                        if input_id in last_write_to_resource:
                            last_write_to_resource[input_id] = op
                ops_which_must_run = set([op])
                continue
            resource_inputs = set()
            for inp, resource_type in _get_resource_inputs(op):
                is_read = resource_type == ResourceType.READ_ONLY
                input_id = ops.tensor_id(inp)
                if input_id in resource_inputs:
                    continue
                resource_inputs.add(input_id)
                if inp.op.type == 'Switch':
                    self._process_switch(inp.op, ops_which_must_run, last_write_to_resource, merge_for_resource)
                is_building_function = op.graph.building_function
                if input_id in last_write_to_resource:
                    if is_building_function or last_write_to_resource[input_id]._control_flow_context is op._control_flow_context:
                        control_inputs.add(last_write_to_resource[input_id])
                if input_id in merge_for_resource:
                    merge_for_resource[input_id]._add_control_input(op)
                if is_read:
                    reads_since_last_write_to_resource[input_id].append(op)
                else:
                    control_inputs.update(reads_since_last_write_to_resource[input_id])
                    reads_since_last_write_to_resource[input_id] = []
                    last_write_to_resource[input_id] = op
            if op_is_stateful(op) and (not resource_inputs) and (op._control_flow_context is None):
                if None in last_write_to_resource:
                    op._add_control_input(last_write_to_resource[None])
                last_write_to_resource[None] = op
            manager_ids = collective_manager_ids_from_op(op)
            for manager_id in manager_ids:
                if manager_id in collective_manager_scopes_opened:
                    op._add_control_input(collective_manager_scopes_opened[manager_id])
                    collective_manager_scopes_opened[manager_id] = op
                else:
                    if manager_id in collective_manager_scopes_used:
                        op._add_control_input(collective_manager_scopes_used[manager_id])
                    collective_manager_scopes_used[manager_id] = op
            if control_inputs and (not is_building_function):
                control_inputs = [c for c in control_inputs if c._control_flow_context is op._control_flow_context]
            op._add_control_inputs(control_inputs)
        self.ops_which_must_run.update(ops_which_must_run)
        control_output_op = None
        for idx, r in enumerate(nest.flatten(list(self._returned_tensors), expand_composites=True)):
            if self.ops_which_must_run:
                updated_ops_which_must_run = []
                if r.graph.building_function:
                    if idx == 0:
                        control_output_op = control_flow_ops.no_op()
                        control_output_op._add_control_inputs(self.ops_which_must_run)
                    updated_ops_which_must_run = [control_output_op]
                else:
                    updated_ops_which_must_run = [o for o in self.ops_which_must_run if o._control_flow_context is r.op._control_flow_context]
                r.op._add_control_inputs(updated_ops_which_must_run)
        self.collective_manager_ids_used = collective_manager_scopes_used