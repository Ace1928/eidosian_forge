import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
class _Sparse_Matrix_coo:

    def __init__(self, vals, rows, cols, shape):
        """
        Create a sparse matrix in coo format.
        *vals*: arrays of values of non-null entries of the matrix
        *rows*: int arrays of rows of non-null entries of the matrix
        *cols*: int arrays of cols of non-null entries of the matrix
        *shape*: 2-tuple (n, m) of matrix shape
        """
        self.n, self.m = shape
        self.vals = np.asarray(vals, dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32)

    def dot(self, V):
        """
        Dot product of self by a vector *V* in sparse-dense to dense format
        *V* dense vector of shape (self.m,).
        """
        assert V.shape == (self.m,)
        return np.bincount(self.rows, weights=self.vals * V[self.cols], minlength=self.m)

    def compress_csc(self):
        """
        Compress rows, cols, vals / summing duplicates. Sort for csc format.
        """
        _, unique, indices = np.unique(self.rows + self.n * self.cols, return_index=True, return_inverse=True)
        self.rows = self.rows[unique]
        self.cols = self.cols[unique]
        self.vals = np.bincount(indices, weights=self.vals)

    def compress_csr(self):
        """
        Compress rows, cols, vals / summing duplicates. Sort for csr format.
        """
        _, unique, indices = np.unique(self.m * self.rows + self.cols, return_index=True, return_inverse=True)
        self.rows = self.rows[unique]
        self.cols = self.cols[unique]
        self.vals = np.bincount(indices, weights=self.vals)

    def to_dense(self):
        """
        Return a dense matrix representing self, mainly for debugging purposes.
        """
        ret = np.zeros([self.n, self.m], dtype=np.float64)
        nvals = self.vals.size
        for i in range(nvals):
            ret[self.rows[i], self.cols[i]] += self.vals[i]
        return ret

    def __str__(self):
        return self.to_dense().__str__()

    @property
    def diag(self):
        """Return the (dense) vector of the diagonal elements."""
        in_diag = self.rows == self.cols
        diag = np.zeros(min(self.n, self.n), dtype=np.float64)
        diag[self.rows[in_diag]] = self.vals[in_diag]
        return diag