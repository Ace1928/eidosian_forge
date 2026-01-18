import copy
from tensorflow.python import tf2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _experimental_run_steps_on_iterator(self, fn, iterator, iterations, initial_loop_values=None):
    if initial_loop_values is None:
        initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)
    ctx = input_lib.MultiStepContext()

    def body(i, *args):
        """A wrapper around `fn` to create the while loop body."""
        del args
        fn_result = fn(ctx, iterator.get_next())
        for name, output in ctx.last_step_outputs.items():
            ctx.last_step_outputs[name] = self._local_results(output)
        flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
        with ops.control_dependencies([fn_result]):
            return [i + 1] + flat_last_step_outputs
    self._outer_control_flow_context = ops.get_default_graph()._get_control_flow_context()
    cond = lambda i, *args: i < iterations
    i = constant_op.constant(0)
    loop_result = while_loop.while_loop(cond, body, [i] + initial_loop_values, name='', parallel_iterations=1, back_prop=False, swap_memory=False, return_same_structure=True)
    del self._outer_control_flow_context
    ctx.run_op = control_flow_ops.group(loop_result)
    last_step_tensor_outputs = loop_result[1:]
    last_step_tensor_outputs_dict = nest.pack_sequence_as(ctx.last_step_outputs, last_step_tensor_outputs)
    for name, reduce_op in ctx._last_step_outputs_reduce_ops.items():
        output = last_step_tensor_outputs_dict[name]
        if reduce_op is None:
            last_step_tensor_outputs_dict[name] = distribute_utils.regroup(output)
        else:
            assert len(output) == 1
            last_step_tensor_outputs_dict[name] = output[0]
    ctx._set_last_step_outputs(last_step_tensor_outputs_dict)
    return ctx