import collections
import contextlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _add_control_dependencies_to_lock(self, created_ops, lock_op):
    """To avoid deadlocks, all args must be executed before lock_op."""
    all_args = set([input_.op for op in created_ops for input_ in op.inputs])
    all_args.update((input_op for op in created_ops for input_op in op.control_inputs))
    all_args_dict = dict(((op._id, op) for op in all_args))
    for op in created_ops:
        all_args_dict.pop(op._id, None)
    for op in lock_op.control_inputs:
        all_args_dict.pop(op._id, None)
    for input_ in lock_op.inputs:
        all_args_dict.pop(input_.op._id, None)
    all_args_dict.pop(lock_op._id, None)
    all_args = all_args_dict.values()
    if not all_args:
        return
    all_args = control_flow_ops.group(*all_args)
    lock_op._add_control_input(all_args)