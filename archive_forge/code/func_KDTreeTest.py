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
def KDTreeTest(kls):
    """Class decorator to create test cases for KDTree and cKDTree

    Tests use the class variable ``kdtree_type`` as the tree constructor.
    """
    if not kls.__name__.startswith('_Test'):
        raise RuntimeError('Expected a class name starting with _Test')
    for tree in (KDTree, cKDTree):
        test_name = kls.__name__[1:] + '_' + tree.__name__
        if test_name in globals():
            raise RuntimeError('Duplicated test name: ' + test_name)
        test_case = type(test_name, (kls,), {'kdtree_type': tree})
        globals()[test_name] = test_case
    return kls