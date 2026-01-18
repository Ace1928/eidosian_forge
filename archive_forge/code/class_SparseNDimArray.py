from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray
from sympy.utilities.iterables import flatten
import functools
class SparseNDimArray(NDimArray):

    def __new__(self, *args, **kwargs):
        return ImmutableSparseNDimArray(*args, **kwargs)

    def __getitem__(self, index):
        """
        Get an element from a sparse N-dim array.

        Examples
        ========

        >>> from sympy import MutableSparseNDimArray
        >>> a = MutableSparseNDimArray(range(4), (2, 2))
        >>> a
        [[0, 1], [2, 3]]
        >>> a[0, 0]
        0
        >>> a[1, 1]
        3
        >>> a[0]
        [0, 1]
        >>> a[1]
        [2, 3]

        Symbolic indexing:

        >>> from sympy.abc import i, j
        >>> a[i, j]
        [[0, 1], [2, 3]][i, j]

        Replace `i` and `j` to get element `(0, 0)`:

        >>> a[i, j].subs({i: 0, j: 0})
        0

        """
        syindex = self._check_symbolic_index(index)
        if syindex is not None:
            return syindex
        index = self._check_index_for_getitem(index)
        if isinstance(index, tuple) and any((isinstance(i, slice) for i in index)):
            sl_factors, eindices = self._get_slice_data_for_array_access(index)
            array = [self._sparse_array.get(self._parse_index(i), S.Zero) for i in eindices]
            nshape = [len(el) for i, el in enumerate(sl_factors) if isinstance(index[i], slice)]
            return type(self)(array, nshape)
        else:
            index = self._parse_index(index)
            return self._sparse_array.get(index, S.Zero)

    @classmethod
    def zeros(cls, *shape):
        """
        Return a sparse N-dim array of zeros.
        """
        return cls({}, shape)

    def tomatrix(self):
        """
        Converts MutableDenseNDimArray to Matrix. Can convert only 2-dim array, else will raise error.

        Examples
        ========

        >>> from sympy import MutableSparseNDimArray
        >>> a = MutableSparseNDimArray([1 for i in range(9)], (3, 3))
        >>> b = a.tomatrix()
        >>> b
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
        """
        from sympy.matrices import SparseMatrix
        if self.rank() != 2:
            raise ValueError('Dimensions must be of size of 2')
        mat_sparse = {}
        for key, value in self._sparse_array.items():
            mat_sparse[self._get_tuple_index(key)] = value
        return SparseMatrix(self.shape[0], self.shape[1], mat_sparse)

    def reshape(self, *newshape):
        new_total_size = functools.reduce(lambda x, y: x * y, newshape)
        if new_total_size != self._loop_size:
            raise ValueError('Invalid reshape parameters ' + newshape)
        return type(self)(self._sparse_array, newshape)