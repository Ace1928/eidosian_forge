import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
class TestDpotr:

    def test_gh_2691(self):
        for lower in [True, False]:
            for clean in [True, False]:
                np.random.seed(42)
                x = np.random.normal(size=(3, 3))
                a = x.dot(x.T)
                dpotrf, dpotri = get_lapack_funcs(('potrf', 'potri'), (a,))
                c, info = dpotrf(a, lower, clean=clean)
                dpt = dpotri(c, lower)[0]
                if lower:
                    assert_allclose(np.tril(dpt), np.tril(inv(a)))
                else:
                    assert_allclose(np.triu(dpt), np.triu(inv(a)))