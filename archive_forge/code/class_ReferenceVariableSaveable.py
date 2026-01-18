import functools
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
class ReferenceVariableSaveable(saveable_object.SaveableObject):
    """SaveableObject implementation that handles reference variables."""

    def __init__(self, var, slice_spec, name):
        spec = saveable_object.SaveSpec(var, slice_spec, name, dtype=var.dtype)
        super(ReferenceVariableSaveable, self).__init__(var, [spec], name)

    def restore(self, restored_tensors, restored_shapes):
        restored_tensor = restored_tensors[0]
        if restored_shapes is not None:
            restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
        return state_ops.assign(self.op, restored_tensor, validate_shape=restored_shapes is None and self.op.get_shape().is_fully_defined())