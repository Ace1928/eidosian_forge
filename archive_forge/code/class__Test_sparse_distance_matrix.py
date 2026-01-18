import os
from numpy.testing import (assert_equal, assert_array_equal, assert_,
from pytest import raises as assert_raises
import pytest
from platform import python_implementation
import numpy as np
from scipy.spatial import KDTree, Rectangle, distance_matrix, cKDTree
from scipy.spatial._ckdtree import cKDTreeNode
from scipy.spatial import minkowski_distance
import itertools
@KDTreeTest
class _Test_sparse_distance_matrix(sparse_distance_matrix_consistency):

    def setup_method(self):
        n = 50
        m = 4
        np.random.seed(1234)
        data1 = np.random.randn(n, m)
        data2 = np.random.randn(n, m)
        self.T1 = self.kdtree_type(data1, leafsize=2)
        self.T2 = self.kdtree_type(data2, leafsize=2)
        self.r = 0.5
        self.p = 2
        self.data1 = data1
        self.data2 = data2
        self.n = n
        self.m = m