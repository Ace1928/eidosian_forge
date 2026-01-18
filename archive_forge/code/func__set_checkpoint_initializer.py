from collections import abc
import os
import time
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export
def _set_checkpoint_initializer(variable, ckpt_file, tensor_name, slice_spec, name='checkpoint_initializer'):
    """Overrides given variable's initialization op.

  Sets variable initializer to assign op that initializes variable from tensor's
  value in the checkpoint.

  Args:
    variable: `tf.Variable` object.
    ckpt_file: string, full path of the checkpoint.
    tensor_name: Name of the tensor to load from the checkpoint.
    slice_spec: Slice specification for loading partitioned tensors.
    name: Name of the operation.
  """
    base_type = variable.dtype.base_dtype
    with ops.device(variable.device), ops.device('/cpu:0'):
        restore_op = io_ops.restore_v2(ckpt_file, [tensor_name], [slice_spec], [base_type], name=name)[0]
        names_to_saveables = saveable_object_util.op_list_to_dict([variable])
        saveable_objects = []
        for name, op in names_to_saveables.items():
            for s in saveable_object_util.saveable_objects_for_op(op, name):
                saveable_objects.append(s)
        assert len(saveable_objects) == 1
    init_op = saveable_objects[0].restore([restore_op], restored_shapes=None)
    variable._initializer_op = init_op
    restore_op.set_shape(variable.shape)
    variable._initial_value = restore_op