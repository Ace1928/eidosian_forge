import functools
import sys
import math
import warnings
import numpy as np
from .._utils import set_module
import numpy.core.numeric as _nx
from numpy.core.numeric import ScalarType, array
from numpy.core.numerictypes import issubdtype
import numpy.matrixlib as matrixlib
from .function_base import diff
from numpy.core.multiarray import ravel_multi_index, unravel_index
from numpy.core import overrides, linspace
from numpy.lib.stride_tricks import as_strided
@set_module('numpy')
class ndindex:
    """
    An N-dimensional iterator object to index arrays.

    Given the shape of an array, an `ndindex` instance iterates over
    the N-dimensional index of the array. At each iteration a tuple
    of indices is returned, the last dimension is iterated over first.

    Parameters
    ----------
    shape : ints, or a single tuple of ints
        The size of each dimension of the array can be passed as
        individual parameters or as the elements of a tuple.

    See Also
    --------
    ndenumerate, flatiter

    Examples
    --------
    Dimensions as individual arguments

    >>> for index in np.ndindex(3, 2, 1):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    Same dimensions - but in a tuple ``(3, 2, 1)``

    >>> for index in np.ndindex((3, 2, 1)):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    """

    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        x = as_strided(_nx.zeros(1), shape=shape, strides=_nx.zeros_like(shape))
        self._it = _nx.nditer(x, flags=['multi_index', 'zerosize_ok'], order='C')

    def __iter__(self):
        return self

    def ndincr(self):
        """
        Increment the multi-dimensional index by one.

        This method is for backward compatibility only: do not use.

        .. deprecated:: 1.20.0
            This method has been advised against since numpy 1.8.0, but only
            started emitting DeprecationWarning as of this version.
        """
        warnings.warn('`ndindex.ndincr()` is deprecated, use `next(ndindex)` instead', DeprecationWarning, stacklevel=2)
        next(self)

    def __next__(self):
        """
        Standard iterator method, updates the index and returns the index
        tuple.

        Returns
        -------
        val : tuple of ints
            Returns a tuple containing the indices of the current
            iteration.

        """
        next(self._it)
        return self._it.multi_index