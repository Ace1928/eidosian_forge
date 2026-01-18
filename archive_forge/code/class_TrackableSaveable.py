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
class TrackableSaveable(saveable_object.SaveableObject):
    """A SaveableObject that defines `Trackable` checkpointing steps."""

    def __init__(self, obj, specs, name, local_names, prefix, call_with_mapped_captures=None):
        self._prefix = prefix
        self._local_names = local_names
        self._trackable = obj
        self._call_with_mapped_captures = call_with_mapped_captures
        super(TrackableSaveable, self).__init__(obj, specs, name)

    def restore(self, restored_tensors, restored_shapes):
        del restored_shapes
        restored_tensor_dict = {}
        for n, local_name in enumerate(self._local_names):
            restored_tensor_dict[local_name] = restored_tensors[n]
        restore_fn = self._trackable._restore_from_tensors
        if not ops.executing_eagerly_outside_functions() and any([spec._tensor.op.type in _REF_VARIABLE_OPS for spec in self.specs if isinstance(spec._tensor, tensor_lib.Tensor)]):
            return restore_fn(restored_tensor_dict)
        if self._call_with_mapped_captures and isinstance(restore_fn, core.ConcreteFunction):
            ret = self._call_with_mapped_captures(restore_fn, [restored_tensor_dict])
        else:
            ret = restore_fn(restored_tensor_dict)
        if ret is not None:
            return ret
        return gen_control_flow_ops.no_op()

    def get_proto_names_and_checkpoint_keys(self):
        return [(self._prefix + local_name, spec.name) for local_name, spec in zip(self._local_names, self.specs)]