from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _tensors_in_key_list(key_list):
    """Generates all Tensors in the given slice spec."""
    if isinstance(key_list, tensor_lib.Tensor):
        yield key_list
    if isinstance(key_list, (list, tuple)):
        for v in key_list:
            for tensor in _tensors_in_key_list(v):
                yield tensor
    if isinstance(key_list, slice):
        for tensor in _tensors_in_key_list(key_list.start):
            yield tensor
        for tensor in _tensors_in_key_list(key_list.stop):
            yield tensor
        for tensor in _tensors_in_key_list(key_list.step):
            yield tensor