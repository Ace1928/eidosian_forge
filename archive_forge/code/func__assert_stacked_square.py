import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def _assert_stacked_square(*arrays):
    """Assert that stacked matrices are square matrices

    Precondition: `arrays` are at least 2d. The caller should assert it
    beforehand. For example,

    >>> def det(a):
    ...     _assert_stacked_2d(a)
    ...     _assert_stacked_square(a)
    ...     ...

    """
    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise linalg.LinAlgError('Last 2 dimensions of the array must be square')