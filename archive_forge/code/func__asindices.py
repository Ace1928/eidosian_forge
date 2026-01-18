import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _asindices(self, idx, length):
    """Convert `idx` to a valid index for an axis with a given length.
        Subclasses that need special validation can override this method.

        idx is assumed to be at least a 1-dimensional array-like, but can
        have no more than 2 dimensions.
        """
    try:
        x = cupy.asarray(idx, dtype=self.indices.dtype)
    except (ValueError, TypeError, MemoryError):
        raise IndexError('invalid index')
    if x.ndim not in (1, 2):
        raise IndexError('Index dimension must be <= 2')
    return x % length