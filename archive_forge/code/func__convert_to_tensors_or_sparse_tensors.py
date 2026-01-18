from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_set_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _convert_to_tensors_or_sparse_tensors(a, b):
    """Convert to tensor types, and flip order if necessary.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`.
    b: `Tensor` or `SparseTensor` of the same type as `a`.

  Returns:
    Tuple of `(a, b, flipped)`, where `a` and `b` have been converted to
    `Tensor` or `SparseTensor`, and `flipped` indicates whether the order has
    been flipped to make it dense,sparse instead of sparse,dense (since the set
    ops do not support the latter).
  """
    a = sparse_tensor.convert_to_tensor_or_sparse_tensor(a, name='a')
    if a.dtype.base_dtype not in _VALID_DTYPES:
        raise TypeError(f"'a' has invalid dtype `{a.dtype}` not in supported dtypes: `{_VALID_DTYPES}`.")
    b = sparse_tensor.convert_to_tensor_or_sparse_tensor(b, name='b')
    if b.dtype.base_dtype != a.dtype.base_dtype:
        raise TypeError("Types don't match, %s vs %s." % (a.dtype, b.dtype))
    if isinstance(a, sparse_tensor.SparseTensor) and (not isinstance(b, sparse_tensor.SparseTensor)):
        return (b, a, True)
    return (a, b, False)