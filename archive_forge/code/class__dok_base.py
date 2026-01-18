import itertools
import numpy as np
from ._matrix import spmatrix
from ._base import _spbase, sparray, issparse
from ._index import IndexMixin
from ._sputils import (isdense, getdtype, isshape, isintlike, isscalarlike,
class _dok_base(_spbase, IndexMixin):
    _format = 'dok'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _spbase.__init__(self)
        self._dict = {}
        self.dtype = getdtype(dtype, default=float)
        if isinstance(arg1, tuple) and isshape(arg1):
            M, N = arg1
            self._shape = check_shape((M, N))
        elif issparse(arg1):
            if arg1.format == self.format and copy:
                arg1 = arg1.copy()
            else:
                arg1 = arg1.todok()
            if dtype is not None:
                arg1 = arg1.astype(dtype, copy=False)
            self._dict.update(arg1)
            self._shape = check_shape(arg1.shape)
            self.dtype = arg1.dtype
        else:
            try:
                arg1 = np.asarray(arg1)
            except Exception as e:
                raise TypeError('Invalid input format.') from e
            if len(arg1.shape) != 2:
                raise TypeError('Expected rank <=2 dense array or matrix.')
            d = self._coo_container(arg1, dtype=dtype).todok()
            self._dict.update(d)
            self._shape = check_shape(arg1.shape)
            self.dtype = d.dtype

    def update(self, val):
        raise NotImplementedError('Direct modification to dok_array element is not allowed.')

    def _update(self, data):
        """An update method for dict data defined for direct access to
        `dok_array` data. Main purpose is to be used for efficient conversion
        from other _spbase classes. Has no checking if `data` is valid."""
        return self._dict.update(data)

    def _getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError('_getnnz over an axis is not implemented for DOK format.')
        return len(self._dict)

    def count_nonzero(self):
        return sum((x != 0 for x in self.values()))
    _getnnz.__doc__ = _spbase._getnnz.__doc__
    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def setdefault(self, key, default=None, /):
        return self._dict.setdefault(key, default)

    def __delitem__(self, key, /):
        del self._dict[key]

    def clear(self):
        return self._dict.clear()

    def popitem(self):
        return self._dict.popitem()

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def get(self, key, default=0.0):
        """This overrides the dict.get method, providing type checking
        but otherwise equivalent functionality.
        """
        try:
            i, j = key
            assert isintlike(i) and isintlike(j)
        except (AssertionError, TypeError, ValueError) as e:
            raise IndexError('Index must be a pair of integers.') from e
        if i < 0 or i >= self.shape[0] or j < 0 or (j >= self.shape[1]):
            raise IndexError('Index out of bounds.')
        return self._dict.get(key, default)

    def _get_intXint(self, row, col):
        return self._dict.get((row, col), self.dtype.type(0))

    def _get_intXslice(self, row, col):
        return self._get_sliceXslice(slice(row, row + 1), col)

    def _get_sliceXint(self, row, col):
        return self._get_sliceXslice(row, slice(col, col + 1))

    def _get_sliceXslice(self, row, col):
        row_start, row_stop, row_step = row.indices(self.shape[0])
        col_start, col_stop, col_step = col.indices(self.shape[1])
        row_range = range(row_start, row_stop, row_step)
        col_range = range(col_start, col_stop, col_step)
        shape = (len(row_range), len(col_range))
        if len(self) >= 2 * shape[0] * shape[1]:
            return self._get_columnXarray(row_range, col_range)
        newdok = self._dok_container(shape, dtype=self.dtype)
        for key in self.keys():
            i, ri = divmod(int(key[0]) - row_start, row_step)
            if ri != 0 or i < 0 or i >= shape[0]:
                continue
            j, rj = divmod(int(key[1]) - col_start, col_step)
            if rj != 0 or j < 0 or j >= shape[1]:
                continue
            newdok._dict[i, j] = self._dict[key]
        return newdok

    def _get_intXarray(self, row, col):
        col = col.squeeze()
        return self._get_columnXarray([row], col)

    def _get_arrayXint(self, row, col):
        row = row.squeeze()
        return self._get_columnXarray(row, [col])

    def _get_sliceXarray(self, row, col):
        row = list(range(*row.indices(self.shape[0])))
        return self._get_columnXarray(row, col)

    def _get_arrayXslice(self, row, col):
        col = list(range(*col.indices(self.shape[1])))
        return self._get_columnXarray(row, col)

    def _get_columnXarray(self, row, col):
        newdok = self._dok_container((len(row), len(col)), dtype=self.dtype)
        for i, r in enumerate(row):
            for j, c in enumerate(col):
                v = self._dict.get((r, c), 0)
                if v:
                    newdok._dict[i, j] = v
        return newdok

    def _get_arrayXarray(self, row, col):
        i, j = map(np.atleast_2d, np.broadcast_arrays(row, col))
        newdok = self._dok_container(i.shape, dtype=self.dtype)
        for key in itertools.product(range(i.shape[0]), range(i.shape[1])):
            v = self._dict.get((i[key], j[key]), 0)
            if v:
                newdok._dict[key] = v
        return newdok

    def _set_intXint(self, row, col, x):
        key = (row, col)
        if x:
            self._dict[key] = x
        elif key in self._dict:
            del self._dict[key]

    def _set_arrayXarray(self, row, col, x):
        row = list(map(int, row.ravel()))
        col = list(map(int, col.ravel()))
        x = x.ravel()
        self._dict.update(zip(zip(row, col), x))
        for i in np.nonzero(x == 0)[0]:
            key = (row[i], col[i])
            if self._dict[key] == 0:
                del self._dict[key]

    def __add__(self, other):
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = self._dok_container(self.shape, dtype=res_dtype)
            M, N = self.shape
            for key in itertools.product(range(M), range(N)):
                aij = self._dict.get(key, 0) + other
                if aij:
                    new[key] = aij
        elif issparse(other):
            if other.format == 'dok':
                if other.shape != self.shape:
                    raise ValueError('Matrix dimensions are not equal.')
                res_dtype = upcast(self.dtype, other.dtype)
                new = self._dok_container(self.shape, dtype=res_dtype)
                new._dict.update(self._dict)
                with np.errstate(over='ignore'):
                    new._dict.update(((k, new[k] + other[k]) for k in other.keys()))
            else:
                csc = self.tocsc()
                new = csc + other
        elif isdense(other):
            new = self.todense() + other
        else:
            return NotImplemented
        return new

    def __radd__(self, other):
        if isscalarlike(other):
            new = self._dok_container(self.shape, dtype=self.dtype)
            M, N = self.shape
            for key in itertools.product(range(M), range(N)):
                aij = self._dict.get(key, 0) + other
                if aij:
                    new[key] = aij
        elif issparse(other):
            if other.format == 'dok':
                if other.shape != self.shape:
                    raise ValueError('Matrix dimensions are not equal.')
                new = self._dok_container(self.shape, dtype=self.dtype)
                new._dict.update(self._dict)
                new._dict.update(((k, self[k] + other[k]) for k in other))
            else:
                csc = self.tocsc()
                new = csc + other
        elif isdense(other):
            new = other + self.todense()
        else:
            return NotImplemented
        return new

    def __neg__(self):
        if self.dtype.kind == 'b':
            raise NotImplementedError('Negating a sparse boolean matrix is not supported.')
        new = self._dok_container(self.shape, dtype=self.dtype)
        new._dict.update(((k, -self[k]) for k in self.keys()))
        return new

    def _mul_scalar(self, other):
        res_dtype = upcast_scalar(self.dtype, other)
        new = self._dok_container(self.shape, dtype=res_dtype)
        new._dict.update(((k, v * other) for k, v in self.items()))
        return new

    def _mul_vector(self, other):
        result = np.zeros(self.shape[0], dtype=upcast(self.dtype, other.dtype))
        for (i, j), v in self.items():
            result[i] += v * other[j]
        return result

    def _mul_multivector(self, other):
        result_shape = (self.shape[0], other.shape[1])
        result_dtype = upcast(self.dtype, other.dtype)
        result = np.zeros(result_shape, dtype=result_dtype)
        for (i, j), v in self.items():
            result[i, :] += v * other[j, :]
        return result

    def __imul__(self, other):
        if isscalarlike(other):
            self._dict.update(((k, v * other) for k, v in self.items()))
            return self
        return NotImplemented

    def __truediv__(self, other):
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = self._dok_container(self.shape, dtype=res_dtype)
            new._dict.update(((k, v / other) for k, v in self.items()))
            return new
        return self.tocsr() / other

    def __itruediv__(self, other):
        if isscalarlike(other):
            self._dict.update(((k, v / other) for k, v in self.items()))
            return self
        return NotImplemented

    def __reduce__(self):
        return dict.__reduce__(self)

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse arrays/matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
        M, N = self.shape
        new = self._dok_container((N, M), dtype=self.dtype, copy=copy)
        new._dict.update((((right, left), val) for (left, right), val in self.items()))
        return new
    transpose.__doc__ = _spbase.transpose.__doc__

    def conjtransp(self):
        """Return the conjugate transpose."""
        M, N = self.shape
        new = self._dok_container((N, M), dtype=self.dtype)
        new._dict.update((((right, left), np.conj(val)) for (left, right), val in self.items()))
        return new

    def copy(self):
        new = self._dok_container(self.shape, dtype=self.dtype)
        new._dict.update(self._dict)
        return new
    copy.__doc__ = _spbase.copy.__doc__

    def tocoo(self, copy=False):
        if self.nnz == 0:
            return self._coo_container(self.shape, dtype=self.dtype)
        idx_dtype = self._get_index_dtype(maxval=max(self.shape))
        data = np.fromiter(self.values(), dtype=self.dtype, count=self.nnz)
        row = np.fromiter((i for i, _ in self.keys()), dtype=idx_dtype, count=self.nnz)
        col = np.fromiter((j for _, j in self.keys()), dtype=idx_dtype, count=self.nnz)
        A = self._coo_container((data, (row, col)), shape=self.shape, dtype=self.dtype)
        A.has_canonical_format = True
        return A
    tocoo.__doc__ = _spbase.tocoo.__doc__

    def todok(self, copy=False):
        if copy:
            return self.copy()
        return self
    todok.__doc__ = _spbase.todok.__doc__

    def tocsc(self, copy=False):
        return self.tocoo(copy=False).tocsc(copy=copy)
    tocsc.__doc__ = _spbase.tocsc.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        newM, newN = shape
        M, N = self.shape
        if newM < M or newN < N:
            for i, j in list(self.keys()):
                if i >= newM or j >= newN:
                    del self._dict[i, j]
        self._shape = shape
    resize.__doc__ = _spbase.resize.__doc__