import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _sputils
class _minmax_mixin(object):
    """Mixin for min and max methods.
    These are not implemented for dia_matrix, hence the separate class.

    """

    def _min_or_max_axis(self, axis, min_or_max, explicit):
        N = self.shape[axis]
        if N == 0:
            raise ValueError('zero-size array to reduction operation')
        M = self.shape[1 - axis]
        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()
        value = mat._minor_reduce(min_or_max, axis, explicit)
        major_index = cupy.arange(M)
        mask = value != 0
        major_index = cupy.compress(mask, major_index)
        value = cupy.compress(mask, value)
        if axis == 0:
            return _coo.coo_matrix((value, (cupy.zeros(len(value)), major_index)), dtype=self.dtype, shape=(1, M))
        else:
            return _coo.coo_matrix((value, (major_index, cupy.zeros(len(value)))), dtype=self.dtype, shape=(M, 1))

    def _min_or_max(self, axis, out, min_or_max, explicit):
        if out is not None:
            raise ValueError("Sparse matrices do not support an 'out' parameter.")
        _sputils.validateaxis(axis)
        if axis is None:
            if 0 in self.shape:
                raise ValueError('zero-size array to reduction operation')
            zero = cupy.zeros((), dtype=self.dtype)
            if self.nnz == 0:
                return zero
            self.sum_duplicates()
            m = min_or_max(self.data)
            if explicit:
                return m
            if self.nnz != internal.prod(self.shape):
                if min_or_max is cupy.min:
                    m = cupy.minimum(zero, m)
                elif min_or_max is cupy.max:
                    m = cupy.maximum(zero, m)
                else:
                    assert False
            return m
        if axis < 0:
            axis += 2
        return self._min_or_max_axis(axis, min_or_max, explicit)

    def _arg_min_or_max_axis(self, axis, op):
        if self.shape[axis] == 0:
            raise ValueError("Can't apply the operation along a zero-sized dimension.")
        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()
        value = mat._arg_minor_reduce(op, axis)
        if axis == 0:
            return value[None, :]
        else:
            return value[:, None]

    def _arg_min_or_max(self, axis, out, op, compare):
        if out is not None:
            raise ValueError("Sparse matrices do not support an 'out' parameter.")
        _sputils.validateaxis(axis)
        if axis is None:
            if 0 in self.shape:
                raise ValueError("Can't apply the operation to an empty matrix.")
            if self.nnz == 0:
                return 0
            else:
                zero = cupy.asarray(self.dtype.type(0))
                mat = self.tocoo()
                mat.sum_duplicates()
                am = op(mat.data)
                m = mat.data[am]
                return cupy.where(compare(m, zero), mat.row[am] * mat.shape[1] + mat.col[am], _non_zero_cmp(mat, am, zero, m))
        if axis < 0:
            axis += 2
        return self._arg_min_or_max_axis(axis, op)

    def max(self, axis=None, out=None, *, explicit=False):
        """Returns the maximum of the matrix or maximum along an axis.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the sum is computed. The default is to
                compute the maximum over all the matrix elements, returning
                a scalar (i.e. ``axis`` = ``None``).
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except
                for the default value, as this argument is not used.
            explicit (bool): Return the maximum value explicitly specified and
                ignore all implicit zero entries. If the dimension has no
                explicit values, a zero is then returned to indicate that it is
                the only implicit value. This parameter is experimental and may
                change in the future.

        Returns:
            (cupy.ndarray or float): Maximum of ``a``. If ``axis`` is
            ``None``, the result is a scalar value. If ``axis`` is given,
            the result is an array of dimension ``a.ndim - 1``. This
            differs from numpy for computational efficiency.

        .. seealso:: min : The minimum value of a sparse matrix along a given
          axis.
        .. seealso:: numpy.matrix.max : NumPy's implementation of ``max`` for
          matrices

        """
        if explicit:
            api_name = 'explicit of cupyx.scipy.sparse.{}.max'.format(self.__class__.__name__)
            _util.experimental(api_name)
        return self._min_or_max(axis, out, cupy.max, explicit)

    def min(self, axis=None, out=None, *, explicit=False):
        """Returns the minimum of the matrix or maximum along an axis.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the sum is computed. The default is to
                compute the minimum over all the matrix elements, returning
                a scalar (i.e. ``axis`` = ``None``).
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except for
                the default value, as this argument is not used.
            explicit (bool): Return the minimum value explicitly specified and
                ignore all implicit zero entries. If the dimension has no
                explicit values, a zero is then returned to indicate that it is
                the only implicit value. This parameter is experimental and may
                change in the future.

        Returns:
            (cupy.ndarray or float): Minimum of ``a``. If ``axis`` is
            None, the result is a scalar value. If ``axis`` is given, the
            result is an array of dimension ``a.ndim - 1``. This differs
            from numpy for computational efficiency.

        .. seealso:: max : The maximum value of a sparse matrix along a given
          axis.
        .. seealso:: numpy.matrix.min : NumPy's implementation of 'min' for
          matrices

        """
        if explicit:
            api_name = 'explicit of cupyx.scipy.sparse.{}.min'.format(self.__class__.__name__)
            _util.experimental(api_name)
        return self._min_or_max(axis, out, cupy.min, explicit)

    def argmax(self, axis=None, out=None):
        """Returns indices of maximum elements along an axis.

        Implicit zero elements are taken into account. If there are several
        maximum values, the index of the first occurrence is returned. If
        ``NaN`` values occur in the matrix, the output defaults to a zero entry
        for the row/column in which the NaN occurs.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the argmax is computed. If ``None`` (default),
                index of the maximum element in the flatten data is returned.
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except for
                the default value, as this argument is not used.

        Returns:
            (cupy.narray or int): Indices of maximum elements. If array,
            its size along ``axis`` is 1.

        """
        return self._arg_min_or_max(axis, out, cupy.argmax, cupy.greater)

    def argmin(self, axis=None, out=None):
        """
        Returns indices of minimum elements along an axis.

        Implicit zero elements are taken into account. If there are several
        minimum values, the index of the first occurrence is returned. If
        ``NaN`` values occur in the matrix, the output defaults to a zero entry
        for the row/column in which the NaN occurs.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the argmin is computed. If ``None`` (default),
                index of the minimum element in the flatten data is returned.
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except for
                the default value, as this argument is not used.

        Returns:
            (cupy.narray or int): Indices of minimum elements. If matrix,
            its size along ``axis`` is 1.

        """
        return self._arg_min_or_max(axis, out, cupy.argmin, cupy.less)