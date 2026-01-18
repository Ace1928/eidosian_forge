import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
class TestCtor:

    def test_basic(self):
        A = np.array([[1, 2], [3, 4]])
        mA = matrix(A)
        assert_(np.all(mA.A == A))
        B = bmat('A,A;A,A')
        C = bmat([[A, A], [A, A]])
        D = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
        assert_(np.all(B.A == D))
        assert_(np.all(C.A == D))
        E = np.array([[5, 6], [7, 8]])
        AEresult = matrix([[1, 2, 5, 6], [3, 4, 7, 8]])
        assert_(np.all(bmat([A, E]) == AEresult))
        vec = np.arange(5)
        mvec = matrix(vec)
        assert_(mvec.shape == (1, 5))

    def test_exceptions(self):
        assert_raises(ValueError, matrix, 'invalid')

    def test_bmat_nondefault_str(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        Aresult = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
        mixresult = np.array([[1, 2, 5, 6], [3, 4, 7, 8], [5, 6, 1, 2], [7, 8, 3, 4]])
        assert_(np.all(bmat('A,A;A,A') == Aresult))
        assert_(np.all(bmat('A,A;A,A', ldict={'A': B}) == Aresult))
        assert_raises(TypeError, bmat, 'A,A;A,A', gdict={'A': B})
        assert_(np.all(bmat('A,A;A,A', ldict={'A': A}, gdict={'A': B}) == Aresult))
        b2 = bmat('A,B;C,D', ldict={'A': A, 'B': B}, gdict={'C': B, 'D': A})
        assert_(np.all(b2 == mixresult))