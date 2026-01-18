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
def _eager_mode_decorator(f, args, kwargs):
    """Implement custom gradient decorator for eager mode."""
    with record.VariableWatcher() as variable_watcher:
        result, grad_fn = f(*args, **kwargs)
    flat_args = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(args))
    flat_kwargs = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(kwargs))
    all_inputs = flat_args + flat_kwargs
    variables = [v.deref() for v in set((v.ref() for v in variable_watcher.watched_variables())) if all((v.deref() is not i for i in all_inputs))]
    grad_argspec = tf_inspect.getfullargspec(grad_fn)
    if variables and 'variables' not in grad_argspec.args and ('variables' not in grad_argspec.kwonlyargs) and (not grad_argspec.varkw):
        raise TypeError("@tf.custom_gradient grad_fn must accept keyword argument 'variables', since function uses variables: {}".format(variables))
    flat_result = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(result))
    flat_result = [gen_array_ops.identity(x) for x in flat_result]
    input_tensors = [ops.convert_to_tensor(x) for x in flat_args + list(variables)]
    recorded_inputs = input_tensors
    arg_count = len(flat_args)

    def actual_grad_fn(*result_grad_components):
        """Custom grad fn wrapper."""
        result_grads = composite_tensor_gradient.replace_flat_tensors_for_gradients(nest.flatten(result), result_grad_components)
        if not isinstance(result_grads, (list, tuple)):
            result_grads = [result_grads]
        if variables:
            input_grads, variable_grads = grad_fn(*result_grads, variables=variables)
            if len(variable_grads) != len(variables):
                raise ValueError('Must return gradient for each variable from @custom_gradient grad_fn.')
        else:
            input_grads = grad_fn(*result_grads)
            variable_grads = []
        flat_grads = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(input_grads))
        if len(flat_grads) != arg_count:
            raise ValueError(f'custom_gradient function expected to return {arg_count} gradients, but returned {len(flat_grads)} instead.')
        return flat_grads + variable_grads
    record.record_operation(f.__name__, flat_result, recorded_inputs, actual_grad_fn)
    flat_result = composite_tensor_gradient.replace_flat_tensors_for_gradients(nest.flatten(result), flat_result)
    return nest.pack_sequence_as(result, flat_result)