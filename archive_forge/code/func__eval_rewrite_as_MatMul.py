from sympy.core import S
from sympy.core.sympify import _sympify
from sympy.functions import KroneckerDelta
from .matexpr import MatrixExpr
from .special import ZeroMatrix, Identity, OneMatrix
def _eval_rewrite_as_MatMul(self, *args, **kwargs):
    from .matmul import MatMul
    mat, perm, axis = self.args
    deep = kwargs.get('deep', True)
    if deep:
        mat = mat.rewrite(MatMul)
    if axis == 0:
        return MatMul(PermutationMatrix(perm), mat)
    elif axis == 1:
        return MatMul(mat, PermutationMatrix(perm ** (-1)))