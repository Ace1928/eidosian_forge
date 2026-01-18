import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array
class MikotaK(LinearOperator):
    """
    Construct a stiffness matrix in various formats of Mikota pair.

    The stiffness matrix `K` is square real tri-diagonal symmetric
    positive definite with integer entries. 

    Parameters
    ----------
    shape : tuple of int
        The shape of the matrix.
    dtype : dtype
        Numerical type of the array. Default is ``np.int32``.

    Methods
    -------
    toarray()
        Construct a dense array from Mikota data
    tosparse()
        Construct a sparse array from Mikota data
    tobanded()
        The format for banded symmetric matrices,
        i.e., (2, n) ndarray with 2 upper diagonals
        placing the main diagonal at the bottom.
    """

    def __init__(self, shape, dtype=np.int32):
        self.shape = shape
        self.dtype = dtype
        super().__init__(dtype, shape)
        n = shape[0]
        self._diag0 = np.arange(2 * n - 1, 0, -2, dtype=self.dtype)
        self._diag1 = -np.arange(n - 1, 0, -1, dtype=self.dtype)

    def tobanded(self):
        return np.array([np.pad(self._diag1, (1, 0), 'constant'), self._diag0])

    def tosparse(self):
        from scipy.sparse import diags
        return diags([self._diag1, self._diag0, self._diag1], [-1, 0, 1], shape=self.shape, dtype=self.dtype)

    def toarray(self):
        return self.tosparse().toarray()

    def _matvec(self, x):
        """
        Construct matrix-free callable banded-matrix-vector multiplication by
        the Mikota stiffness matrix without constructing or storing the matrix
        itself using the knowledge of its entries and the 3-diagonal format.
        """
        x = x.reshape(self.shape[0], -1)
        result_dtype = np.promote_types(x.dtype, self.dtype)
        kx = np.zeros_like(x, dtype=result_dtype)
        d1 = self._diag1
        d0 = self._diag0
        kx[0, :] = d0[0] * x[0, :] + d1[0] * x[1, :]
        kx[-1, :] = d1[-1] * x[-2, :] + d0[-1] * x[-1, :]
        kx[1:-1, :] = d1[:-1, None] * x[:-2, :] + d0[1:-1, None] * x[1:-1, :] + d1[1:, None] * x[2:, :]
        return kx

    def _matmat(self, x):
        """
        Construct matrix-free callable matrix-matrix multiplication by
        the Stiffness mass matrix without constructing or storing the matrix itself
        by reusing the ``_matvec(x)`` that supports both 1D and 2D arrays ``x``.
        """
        return self._matvec(x)

    def _adjoint(self):
        return self

    def _transpose(self):
        return self