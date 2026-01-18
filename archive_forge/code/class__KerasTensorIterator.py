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
class _KerasTensorIterator(object):
    """Iterates over the leading dim of a KerasTensor. Performs 0 error checks."""

    def __init__(self, tensor, dim0):
        self._tensor = tensor
        self._index = 0
        self._limit = dim0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == self._limit:
            raise StopIteration
        result = self._tensor[self._index]
        self._index += 1
        return result