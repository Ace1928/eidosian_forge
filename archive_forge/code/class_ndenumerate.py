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
class ndenumerate:
    """
    Multidimensional index iterator.

    Return an iterator yielding pairs of array coordinates and values.

    Parameters
    ----------
    arr : ndarray
      Input array.

    See Also
    --------
    ndindex, flatiter

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> for index, x in np.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 1
    (0, 1) 2
    (1, 0) 3
    (1, 1) 4

    """

    def __init__(self, arr):
        self.iter = np.asarray(arr).flat

    def __next__(self):
        """
        Standard iterator method, returns the index tuple and array value.

        Returns
        -------
        coords : tuple of ints
            The indices of the current iteration.
        val : scalar
            The array element of the current iteration.

        """
        return (self.iter.coords, next(self.iter))

    def __iter__(self):
        return self