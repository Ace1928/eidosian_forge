import inspect
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_decorator
def _update_weak_tensor_patched_ops_in_dispatch_dict(patched_op):
    """Update dispatch dictionary to store WeakTensor patched op references.

  _TYPE_BASED_DISPATCH_SIGNATURES in dispatch.py stores mappings from op
  reference to all the dispatchers it's registered with. We need to update
  this dictionary to add a mapping from the patched-op reference to the
  signature dictionary the unpatched-op reference is mapped to. This ensures
  that dispatch can be reigstered and unregistered with monkey-patched ops.
  """
    dispatch_dict = dispatch._TYPE_BASED_DISPATCH_SIGNATURES
    unpatched_api = patched_op.__wrapped__
    if unpatched_api in dispatch_dict:
        dispatch_dict[patched_op] = dispatch_dict[unpatched_api]