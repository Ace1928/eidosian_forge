from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _graph_mode_decorator(f, args, kwargs):
    """Implement custom gradient decorator for graph mode."""
    if kwargs:
        raise ValueError('The custom_gradient decorator currently supports keywords arguments only when eager execution is enabled.')
    name = generate_name()
    args = variable_utils.convert_variables_to_tensors(args)
    args = nest.map_structure(ops.convert_to_tensor, args, expand_composites=True)
    current_var_scope = variable_scope.get_variable_scope()
    before_vars = set([v.ref() for v in current_var_scope.global_variables() + current_var_scope.local_variables()])
    with record.VariableWatcher() as variable_watcher:
        result, grad_fn = f(*args)
    flat_args = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(args))
    flat_result = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(result))
    flat_result_len = len(flat_result)
    after_vars = set([v.ref() for v in current_var_scope.global_variables() + current_var_scope.local_variables()])
    new_vars = after_vars - before_vars
    new_vars_list = [v.deref() for v in new_vars]
    for v in new_vars_list:
        if not resource_variable_ops.is_resource_variable(v):
            raise TypeError('All variables used by a function wrapped with @custom_gradient must be `ResourceVariable`s. Ensure that no `variable_scope` is created with `use_resource=False`.')
    variables_in_tape = frozenset([v.ref() for v in variable_watcher.watched_variables()])
    graphs = {getattr(o, 'graph', None) for o in flat_result}
    graphs.discard(None)
    if graphs:
        if len(graphs) > 1:
            raise ValueError('All custom_gradient outputs should be from the same graph')
        output_graph = graphs.pop()
        filtered_input_tensors = []
        for i in flat_args:
            if i.graph == output_graph:
                filtered_input_tensors.append(i)
    else:
        filtered_input_tensors = flat_args
    variables_in_subgraph = frozenset([v.ref() for v in _get_dependent_variables(input_ops=filtered_input_tensors, output_ops=flat_result)])
    variables = sorted([v.deref() for v in variables_in_subgraph.union(variables_in_tape)], key=lambda v: v.name)
    grad_argspec = tf_inspect.getfullargspec(grad_fn)
    variables_in_signature = 'variables' in grad_argspec.args or 'variables' in grad_argspec.kwonlyargs or grad_argspec.varkw
    if variables and (not variables_in_signature):
        raise TypeError("@tf.custom_gradient grad_fn must accept keyword argument 'variables', since function uses variables: {}".format(variables))
    if variables_in_signature and (not variables):
        logging.vlog(1, "@custom_gradient grad_fn has 'variables' in signature, but no ResourceVariables were used on the forward pass.")
    all_tensors = flat_result + flat_args + variables

    def tape_grad_fn(*result_grad_components):
        """Custom grad fn wrapper."""
        result_grads = composite_tensor_gradient.replace_flat_tensors_for_gradients(nest.flatten(result), result_grad_components[:flat_result_len])
        if not isinstance(result_grads, (list, tuple)):
            result_grads = [result_grads]
        if variables:
            input_grads, variable_grads = grad_fn(*result_grads, variables=variables)
            if len(variable_grads) != len(variables):
                raise ValueError('Must return gradient for each variable from @custom_gradient grad_fn.')
        else:
            input_grads = grad_fn(*result_grads)
            variable_grads = []
        input_grads = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(input_grads))
        return [None] * flat_result_len + input_grads + variable_grads

    @ops.RegisterGradient(name)
    def internal_grad_fn(unused_op, *result_grads):
        """Custom grad fn wrapper."""
        return tape_grad_fn(*result_grads)
    original_tensors = all_tensors
    with ops.get_default_graph().gradient_override_map({'IdentityN': name}):
        all_tensors = array_ops.identity_n(all_tensors)
    original_tensors = [ops.convert_to_tensor(x) for x in original_tensors]
    for i, t in enumerate(original_tensors):
        if t.dtype == dtypes.resource and hasattr(t, '_handle_data'):
            all_tensors[i]._handle_data = t._handle_data
    record.record_operation(f.__name__, all_tensors, original_tensors, tape_grad_fn)
    for ot, t in zip(original_tensors, all_tensors):
        handle_data_util.copy_handle_data(ot, t)
    flat_result = composite_tensor_gradient.replace_flat_tensors_for_gradients(nest.flatten(result), all_tensors[:flat_result_len])
    return nest.pack_sequence_as(result, flat_result)