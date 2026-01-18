from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
class UserRegisteredTypeKerasTensor(KerasTensor):
    """KerasTensor that represents legacy register_symbolic_tensor_type."""

    def __init__(self, user_registered_symbolic_object):
        x = user_registered_symbolic_object
        self._user_registered_symbolic_object = x
        type_spec = UserRegisteredSpec(x.shape, x.dtype)
        name = getattr(x, 'name', None)
        super(UserRegisteredTypeKerasTensor, self).__init__(type_spec, name)

    @classmethod
    def from_tensor(cls, tensor):
        return cls(tensor)

    @classmethod
    def from_type_spec(cls, type_spec, name=None):
        raise NotImplementedError('You cannot instantiate a KerasTensor directly from TypeSpec: %s' % type_spec)

    def _to_placeholder(self):
        return self._user_registered_symbolic_object