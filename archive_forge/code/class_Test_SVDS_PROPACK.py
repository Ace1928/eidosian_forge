import os
import re
import copy
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pytest
from scipy.linalg import svd, null_space
from scipy.sparse import csc_matrix, issparse, spdiags, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg import svds
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence
class Test_SVDS_PROPACK(SVDSCommonTests):

    def setup_method(self):
        self.solver = 'propack'

    def test_svd_LM_ones_matrix(self):
        message = 'PROPACK does not return orthonormal singular vectors associated with zero singular values.'
        pytest.xfail(message)

    def test_svd_LM_zeros_matrix(self):
        message = 'PROPACK does not return orthonormal singular vectors associated with zero singular values.'
        pytest.xfail(message)