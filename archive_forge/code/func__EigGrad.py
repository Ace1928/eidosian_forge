from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('Eig')
def _EigGrad(op, grad_e, grad_v):
    """Gradient for Eig.

  Based on eq. 4.77 from paper by
  Christoph Boeddeker et al.
  https://arxiv.org/abs/1701.00392
  See also
  "Computation of eigenvalue and eigenvector derivatives
  for a general complex-valued eigensystem" by Nico van der Aa.
  As for now only distinct eigenvalue case is considered.
  """
    e = op.outputs[0]
    compute_v = op.get_attr('compute_v')
    with ops.control_dependencies([grad_e, grad_v]):
        if compute_v:
            v = op.outputs[1]
            vt = _linalg.adjoint(v)
            f = array_ops.matrix_set_diag(_SafeReciprocal(array_ops.expand_dims(e, -2) - array_ops.expand_dims(e, -1)), array_ops.zeros_like(e))
            f = math_ops.conj(f)
            vgv = math_ops.matmul(vt, grad_v)
            mid = array_ops.matrix_diag(grad_e)
            diag_grad_part = array_ops.matrix_diag(array_ops.matrix_diag_part(math_ops.cast(math_ops.real(vgv), vgv.dtype)))
            mid += f * (vgv - math_ops.matmul(math_ops.matmul(vt, v), diag_grad_part))
            grad_a = linalg_ops.matrix_solve(vt, math_ops.matmul(mid, vt))
        else:
            _, v = linalg_ops.eig(op.inputs[0])
            vt = _linalg.adjoint(v)
            grad_a = linalg_ops.matrix_solve(vt, math_ops.matmul(array_ops.matrix_diag(grad_e), vt))
        return math_ops.cast(grad_a, op.inputs[0].dtype)