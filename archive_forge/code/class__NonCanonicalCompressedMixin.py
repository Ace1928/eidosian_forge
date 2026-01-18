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
class _NonCanonicalCompressedMixin(_NonCanonicalMixin):

    def _arg1_for_noncanonical(self, M, sorted_indices=False):
        """Return non-canonical constructor arg1 equivalent to M"""
        data, indices, indptr = _same_sum_duplicate(M.data, M.indices, indptr=M.indptr)
        if not sorted_indices:
            for start, stop in zip(indptr, indptr[1:]):
                indices[start:stop] = indices[start:stop][::-1].copy()
                data[start:stop] = data[start:stop][::-1].copy()
        return (data, indices, indptr)

    def _insert_explicit_zero(self, M, i, j):
        M[i, j] = 0
        return M