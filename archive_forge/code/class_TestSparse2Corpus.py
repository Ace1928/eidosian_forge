import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
class TestSparse2Corpus(unittest.TestCase):

    def setUp(self):
        self.orig_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.s2c = matutils.Sparse2Corpus(csc_matrix(self.orig_array))

    def test_getitem_slice(self):
        assert_array_equal(self.s2c[:2].sparse.toarray(), self.orig_array[:, :2])
        assert_array_equal(self.s2c[1:3].sparse.toarray(), self.orig_array[:, 1:3])

    def test_getitem_index(self):
        self.assertListEqual(self.s2c[1], [(0, 2), (1, 5), (2, 8)])

    def test_getitem_list_of_indices(self):
        assert_array_equal(self.s2c[[1, 2]].sparse.toarray(), self.orig_array[:, [1, 2]])
        assert_array_equal(self.s2c[[1]].sparse.toarray(), self.orig_array[:, [1]])

    def test_getitem_ndarray(self):
        assert_array_equal(self.s2c[np.array([1, 2])].sparse.toarray(), self.orig_array[:, [1, 2]])
        assert_array_equal(self.s2c[np.array([1])].sparse.toarray(), self.orig_array[:, [1]])

    def test_getitem_range(self):
        assert_array_equal(self.s2c[range(1, 3)].sparse.toarray(), self.orig_array[:, [1, 2]])
        assert_array_equal(self.s2c[range(1, 2)].sparse.toarray(), self.orig_array[:, [1]])

    def test_getitem_ellipsis(self):
        assert_array_equal(self.s2c[...].sparse.toarray(), self.orig_array)