from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('Qr')
def _QrGrad(op, dq, dr):
    """Gradient for Qr."""
    q, r = op.outputs
    if r.shape.ndims is None or r.shape.as_list()[-2] is None or r.shape.as_list()[-1] is None:
        raise NotImplementedError(f'QrGrad not implemented with dynamic shapes. Received r.shape: {r.shape}')
    if r.shape.dims[-2].value > r.shape.dims[-1].value and q.shape.dims[-2].value == q.shape.dims[-1].value:
        raise NotImplementedError(f'QrGrad not implemented when nrows > ncols and full_matrices is true. Received r.shape={r.shape} with nrows={r.shape.dims[-2]}and ncols={r.shape.dims[-1]}.')

    def _TriangularSolve(x, r):
        """Equiv to matmul(x, adjoint(matrix_inverse(r))) if r is upper-tri."""
        return _linalg.adjoint(linalg_ops.matrix_triangular_solve(r, _linalg.adjoint(x), lower=False, adjoint=False))

    def _QrGradSquareAndDeepMatrices(q, r, dq, dr):
        """Gradient for matrix orders num_rows >= num_cols
    and full_matrices is false.
    """
        qdq = math_ops.matmul(q, dq, adjoint_a=True)
        qdq_ = qdq - _linalg.adjoint(qdq)
        rdr = math_ops.matmul(r, dr, adjoint_b=True)
        rdr_ = rdr - _linalg.adjoint(rdr)
        tril = array_ops.matrix_band_part(qdq_ + rdr_, -1, 0)
        grad_a = math_ops.matmul(q, dr + _TriangularSolve(tril, r))
        grad_b = _TriangularSolve(dq - math_ops.matmul(q, qdq), r)
        ret = grad_a + grad_b
        if q.dtype.is_complex:
            m = rdr - _linalg.adjoint(qdq)
            eyem = _linalg.set_diag(array_ops.zeros_like(m), _linalg.diag_part(m))
            correction = eyem - math_ops.cast(math_ops.real(eyem), q.dtype)
            ret = ret + _TriangularSolve(math_ops.matmul(q, _linalg.adjoint(correction)), r)
        return ret
    num_rows, num_cols = (q.shape.dims[-2].value, r.shape.dims[-1])
    if num_rows >= num_cols:
        return _QrGradSquareAndDeepMatrices(q, r, dq, dr)
    a = op.inputs[0]
    y = a[..., :, num_rows:]
    u = r[..., :, :num_rows]
    dv = dr[..., :, num_rows:]
    du = dr[..., :, :num_rows]
    dy = math_ops.matmul(q, dv)
    dx = _QrGradSquareAndDeepMatrices(q, u, dq + math_ops.matmul(y, dv, adjoint_b=True), du)
    return array_ops.concat([dx, dy], axis=-1)