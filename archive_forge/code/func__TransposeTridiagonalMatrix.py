from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
def _TransposeTridiagonalMatrix(diags):
    """Transposes a tridiagonal matrix.

  Args:
    diags: the diagonals of the input matrix in the compact form (see
      linalg_ops.tridiagonal_solve).

  Returns:
    Diagonals of the transposed matrix in the compact form.
  """
    diag = diags[..., 1, :]
    if diags.shape.is_fully_defined():
        zeros = array_ops.zeros(list(diags.shape[:-2]) + [1], dtype=diags.dtype)
        superdiag = array_ops.concat((diags[..., 2, 1:], zeros), axis=-1)
        subdiag = array_ops.concat((zeros, diags[..., 0, :-1]), axis=-1)
    else:
        rank = array_ops.rank(diags)
        zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
        superdiag_pad = array_ops.concat((zeros, array_ops.constant([[0, 1]])), axis=0)
        superdiag = array_ops.pad(diags[..., 2, 1:], superdiag_pad)
        subdiag_pad = array_ops.concat((zeros, array_ops.constant([[1, 0]])), axis=0)
        subdiag = array_ops.pad(diags[..., 0, :-1], subdiag_pad)
    return array_ops_stack.stack([superdiag, diag, subdiag], axis=-2)