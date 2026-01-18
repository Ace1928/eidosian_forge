import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array
class Sakurai(LinearOperator):
    """
    Construct a Sakurai matrix in various formats and its eigenvalues.

    Constructs the "Sakurai" matrix motivated by reference [1]_:
    square real symmetric positive definite and 5-diagonal
    with the main digonal ``[5, 6, 6, ..., 6, 6, 5], the ``+1`` and ``-1``
    diagonals filled with ``-4``, and the ``+2`` and ``-2`` diagonals
    made of ``1``. Its eigenvalues are analytically known to be
    ``16. * np.power(np.cos(0.5 * k * np.pi / (n + 1)), 4)``.
    The matrix gets ill-conditioned with its size growing.
    It is useful for testing and benchmarking sparse eigenvalue solvers
    especially those taking advantage of its banded 5-diagonal structure.
    See the notes below for details.

    Parameters
    ----------
    n : int
        The size of the matrix.
    dtype : dtype
        Numerical type of the array. Default is ``np.int8``.

    Methods
    -------
    toarray()
        Construct a dense array from Laplacian data
    tosparse()
        Construct a sparse array from Laplacian data
    tobanded()
        The Sakurai matrix in the format for banded symmetric matrices,
        i.e., (3, n) ndarray with 3 upper diagonals
        placing the main diagonal at the bottom.
    eigenvalues
        All eigenvalues of the Sakurai matrix ordered ascending.

    Notes
    -----
    Reference [1]_ introduces a generalized eigenproblem for the matrix pair
    `A` and `B` where `A` is the identity so we turn it into an eigenproblem
    just for the matrix `B` that this function outputs in various formats
    together with its eigenvalues.
    
    .. versionadded:: 1.12.0

    References
    ----------
    .. [1] T. Sakurai, H. Tadano, Y. Inadomi, and U. Nagashima,
       "A moment-based method for large-scale generalized
       eigenvalue problems",
       Appl. Num. Anal. Comp. Math. Vol. 1 No. 2 (2004).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg._special_sparse_arrays import Sakurai
    >>> from scipy.linalg import eig_banded
    >>> n = 6
    >>> sak = Sakurai(n)

    Since all matrix entries are small integers, ``'int8'`` is
    the default dtype for storing matrix representations.

    >>> sak.toarray()
    array([[ 5, -4,  1,  0,  0,  0],
           [-4,  6, -4,  1,  0,  0],
           [ 1, -4,  6, -4,  1,  0],
           [ 0,  1, -4,  6, -4,  1],
           [ 0,  0,  1, -4,  6, -4],
           [ 0,  0,  0,  1, -4,  5]], dtype=int8)
    >>> sak.tobanded()
    array([[ 1,  1,  1,  1,  1,  1],
           [-4, -4, -4, -4, -4, -4],
           [ 5,  6,  6,  6,  6,  5]], dtype=int8)
    >>> sak.tosparse()
    <6x6 sparse matrix of type '<class 'numpy.int8'>'
        with 24 stored elements (5 diagonals) in DIAgonal format>
    >>> np.array_equal(sak.dot(np.eye(n)), sak.tosparse().toarray())
    True
    >>> sak.eigenvalues()
    array([0.03922866, 0.56703972, 2.41789479, 5.97822974,
           10.54287655, 14.45473055])
    >>> sak.eigenvalues(2)
    array([0.03922866, 0.56703972])

    The banded form can be used in scipy functions for banded matrices, e.g.,

    >>> e = eig_banded(sak.tobanded(), eigvals_only=True)
    >>> np.allclose(sak.eigenvalues, e, atol= n * n * n * np.finfo(float).eps)
    True

    """

    def __init__(self, n, dtype=np.int8):
        self.n = n
        self.dtype = dtype
        shape = (n, n)
        super().__init__(dtype, shape)

    def eigenvalues(self, m=None):
        """Return the requested number of eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of smallest eigenvalues to return.
            If not provided, then all eigenvalues will be returned.
            
        Returns
        -------
        eigenvalues : `np.float64` array
            The requested `m` smallest or all eigenvalues, in ascending order.
        """
        if m is None:
            m = self.n
        k = np.arange(self.n + 1 - m, self.n + 1)
        return np.flip(16.0 * np.power(np.cos(0.5 * k * np.pi / (self.n + 1)), 4))

    def tobanded(self):
        """
        Construct the Sakurai matrix as a banded array.
        """
        d0 = np.r_[5, 6 * np.ones(self.n - 2, dtype=self.dtype), 5]
        d1 = -4 * np.ones(self.n, dtype=self.dtype)
        d2 = np.ones(self.n, dtype=self.dtype)
        return np.array([d2, d1, d0]).astype(self.dtype)

    def tosparse(self):
        """
        Construct the Sakurai matrix is a sparse format.
        """
        from scipy.sparse import spdiags
        d = self.tobanded()
        return spdiags([d[0], d[1], d[2], d[1], d[0]], [-2, -1, 0, 1, 2], self.n, self.n)

    def toarray(self):
        return self.tosparse().toarray()

    def _matvec(self, x):
        """
        Construct matrix-free callable banded-matrix-vector multiplication by
        the Sakurai matrix without constructing or storing the matrix itself
        using the knowledge of its entries and the 5-diagonal format.
        """
        x = x.reshape(self.n, -1)
        result_dtype = np.promote_types(x.dtype, self.dtype)
        sx = np.zeros_like(x, dtype=result_dtype)
        sx[0, :] = 5 * x[0, :] - 4 * x[1, :] + x[2, :]
        sx[-1, :] = 5 * x[-1, :] - 4 * x[-2, :] + x[-3, :]
        sx[1:-1, :] = 6 * x[1:-1, :] - 4 * (x[:-2, :] + x[2:, :]) + np.pad(x[:-3, :], ((1, 0), (0, 0))) + np.pad(x[3:, :], ((0, 1), (0, 0)))
        return sx

    def _matmat(self, x):
        """
        Construct matrix-free callable matrix-matrix multiplication by
        the Sakurai matrix without constructing or storing the matrix itself
        by reusing the ``_matvec(x)`` that supports both 1D and 2D arrays ``x``.
        """
        return self._matvec(x)

    def _adjoint(self):
        return self

    def _transpose(self):
        return self