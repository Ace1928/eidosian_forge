import numpy as np
from ._matrix import spmatrix
from ._base import _spbase, sparray
from ._sparsetools import csc_tocsr, expandptr
from ._sputils import upcast
from ._compressed import _cs_matrix
class _csc_base(_cs_matrix):
    _format = 'csc'

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse arrays/matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
        M, N = self.shape
        return self._csr_container((self.data, self.indices, self.indptr), (N, M), copy=copy)
    transpose.__doc__ = _spbase.transpose.__doc__

    def __iter__(self):
        yield from self.tocsr()

    def tocsc(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self
    tocsc.__doc__ = _spbase.tocsc.__doc__

    def tocsr(self, copy=False):
        M, N = self.shape
        idx_dtype = self._get_index_dtype((self.indptr, self.indices), maxval=max(self.nnz, N))
        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))
        csc_tocsr(M, N, self.indptr.astype(idx_dtype), self.indices.astype(idx_dtype), self.data, indptr, indices, data)
        A = self._csr_container((data, indices, indptr), shape=self.shape, copy=False)
        A.has_sorted_indices = True
        return A
    tocsr.__doc__ = _spbase.tocsr.__doc__

    def nonzero(self):
        major_dim, minor_dim = self._swap(self.shape)
        minor_indices = self.indices
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
        expandptr(major_dim, self.indptr, major_indices)
        row, col = self._swap((major_indices, minor_indices))
        nz_mask = self.data != 0
        row = row[nz_mask]
        col = col[nz_mask]
        ind = np.argsort(row, kind='mergesort')
        row = row[ind]
        col = col[ind]
        return (row, col)
    nonzero.__doc__ = _cs_matrix.nonzero.__doc__

    def _getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        return self._get_submatrix(minor=i).tocsr()

    def _getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSC matrix (column vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        return self._get_submatrix(major=i, copy=True)

    def _get_intXarray(self, row, col):
        return self._major_index_fancy(col)._get_submatrix(minor=row)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._major_slice(col)._get_submatrix(minor=row)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._get_submatrix(major=col)._minor_slice(row)

    def _get_sliceXarray(self, row, col):
        return self._major_index_fancy(col)._minor_slice(row)

    def _get_arrayXint(self, row, col):
        return self._get_submatrix(major=col)._minor_index_fancy(row)

    def _get_arrayXslice(self, row, col):
        return self._major_slice(col)._minor_index_fancy(row)

    def _swap(self, x):
        """swap the members of x if this is a column-oriented matrix
        """
        return (x[1], x[0])