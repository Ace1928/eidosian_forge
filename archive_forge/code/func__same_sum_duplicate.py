import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def _same_sum_duplicate(data, *inds, **kwargs):
    """Duplicates entries to produce the same matrix"""
    indptr = kwargs.pop('indptr', None)
    if np.issubdtype(data.dtype, np.bool_) or np.issubdtype(data.dtype, np.unsignedinteger):
        if indptr is None:
            return (data,) + inds
        else:
            return (data,) + inds + (indptr,)
    zeros_pos = (data == 0).nonzero()
    data = data.repeat(2, axis=0)
    data[::2] -= 1
    data[1::2] = 1
    if zeros_pos[0].size > 0:
        pos = tuple((p[0] for p in zeros_pos))
        pos1 = (2 * pos[0],) + pos[1:]
        pos2 = (2 * pos[0] + 1,) + pos[1:]
        data[pos1] = 0
        data[pos2] = 0
    inds = tuple((indices.repeat(2) for indices in inds))
    if indptr is None:
        return (data,) + inds
    else:
        return (data,) + inds + (indptr * 2,)