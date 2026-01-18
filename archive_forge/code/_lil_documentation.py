from bisect import bisect_left
import numpy as np
from ._matrix import spmatrix
from ._base import _spbase, sparray, issparse
from ._index import IndexMixin, INT_TYPES, _broadcast_arrays
from ._sputils import (getdtype, isshape, isscalarlike, upcast_scalar,
from . import _csparsetools

        Fast path for indexing in the case where column index is slice.

        This gains performance improvement over brute force by more
        efficient skipping of zeros, by accessing the elements
        column-wise in order.

        Parameters
        ----------
        rows : sequence or range
            Rows indexed. If range, must be within valid bounds.
        col_slice : slice
            Columns indexed

        