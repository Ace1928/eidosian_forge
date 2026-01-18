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
class SaveableCompatibilityConverter(trackable.Trackable):
    """Converts object's `SaveableObjects` to functions used in TF2 checkpointing.

  A class that converts a Trackable object's `SaveableObjects` to save and
  restore functions with the same signatures as
  `Trackable._serialize_to_tensors` and `Trackable._restore_from_tensors`.
  This class also produces a method for filling the object proto.
  """
    __slots__ = ('_obj', '_saveables')

    def __init__(self, obj, saveables):
        """Constructor.

    Args:
      obj: A Trackable object.
      saveables: A list of saveables for `obj`.
    """
        self._obj = obj
        self._saveables = saveables

    @property
    def obj(self):
        return self._obj

    @property
    def saveables(self):
        """Returns a list of SaveableObjects generated from the Trackable object."""
        return self._saveables

    def _serialize_to_tensors(self):
        """Returns a dict of tensors to serialize."""
        return saveable_object_to_tensor_dict(self.saveables)

    def _restore_from_tensors(self, restored_tensors):
        """Returns the restore ops defined in the Saveables."""
        expected_keys = []
        for saveable in self.saveables:
            expected_keys.extend((trackable_utils.extract_local_name(_convert_to_string(spec.name)) for spec in saveable.specs))
        if set(expected_keys) != restored_tensors.keys():
            raise ValueError(f'Could not restore object {self._obj} because not all expected tensors were in the checkpoint.\n\tExpected: {expected_keys}\n\tGot: {list(restored_tensors.keys())}')
        return saveable_object_to_restore_fn(self.saveables)(restored_tensors)