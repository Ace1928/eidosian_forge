import numpy
import cupy
import cupyx.cusolver
from cupy import cublas
from cupyx import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _interface
from cupyx.scipy.sparse.linalg._iterative import _make_system
import warnings
class SuperLU:

    def __init__(self, obj):
        """LU factorization of a sparse matrix.

        Args:
            obj (scipy.sparse.linalg.SuperLU): LU factorization of a sparse
                matrix, computed by `scipy.sparse.linalg.splu`, etc.
        """
        if not scipy_available:
            raise RuntimeError('scipy is not available')
        if not isinstance(obj, scipy.sparse.linalg.SuperLU):
            raise TypeError('obj must be scipy.sparse.linalg.SuperLU')
        self.shape = obj.shape
        self.nnz = obj.nnz
        self.perm_r = cupy.array(obj.perm_r)
        self.perm_c = cupy.array(obj.perm_c)
        self.L = sparse.csr_matrix(obj.L.tocsr())
        self.U = sparse.csr_matrix(obj.U.tocsr())
        self._perm_r_rev = cupy.argsort(self.perm_r)
        self._perm_c_rev = cupy.argsort(self.perm_c)

    def solve(self, rhs, trans='N'):
        """Solves linear system of equations with one or several right-hand sides.

        Args:
            rhs (cupy.ndarray): Right-hand side(s) of equation with dimension
                ``(M)`` or ``(M, K)``.
            trans (str): 'N', 'T' or 'H'.
                'N': Solves ``A * x = rhs``.
                'T': Solves ``A.T * x = rhs``.
                'H': Solves ``A.conj().T * x = rhs``.

        Returns:
            cupy.ndarray:
                Solution vector(s)
        """
        if not isinstance(rhs, cupy.ndarray):
            raise TypeError('ojb must be cupy.ndarray')
        if rhs.ndim not in (1, 2):
            raise ValueError('rhs.ndim must be 1 or 2 (actual: {})'.format(rhs.ndim))
        if rhs.shape[0] != self.shape[0]:
            raise ValueError('shape mismatch (self.shape: {}, rhs.shape: {})'.format(self.shape, rhs.shape))
        if trans not in ('N', 'T', 'H'):
            raise ValueError("trans must be 'N', 'T', or 'H'")
        if cusparse.check_availability('spsm') and _should_use_spsm(rhs):

            def spsm(A, B, lower, transa):
                return cusparse.spsm(A, B, lower=lower, transa=transa)
            sm = spsm
        elif cusparse.check_availability('csrsm2'):

            def csrsm2(A, B, lower, transa):
                cusparse.csrsm2(A, B, lower=lower, transa=transa)
                return B
            sm = csrsm2
        else:
            raise NotImplementedError
        x = rhs.astype(self.L.dtype)
        if trans == 'N':
            if self.perm_r is not None:
                if x.ndim == 2 and x._f_contiguous:
                    x = x.T[:, self._perm_r_rev].T
                else:
                    x = x[self._perm_r_rev]
            x = sm(self.L, x, lower=True, transa=trans)
            x = sm(self.U, x, lower=False, transa=trans)
            if self.perm_c is not None:
                x = x[self.perm_c]
        else:
            if self.perm_c is not None:
                if x.ndim == 2 and x._f_contiguous:
                    x = x.T[:, self._perm_c_rev].T
                else:
                    x = x[self._perm_c_rev]
            x = sm(self.U, x, lower=False, transa=trans)
            x = sm(self.L, x, lower=True, transa=trans)
            if self.perm_r is not None:
                x = x[self.perm_r]
        if not x._f_contiguous:
            x = x.copy(order='F')
        return x