import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestNonIntegerArrayLike:
    """Tests that array_likes only valid if can safely cast to integer.

    For instance, lists give IndexError when they cannot be safely cast to
    an integer.

    """

    def test_basic(self):
        a = np.arange(10)
        assert_raises(IndexError, a.__getitem__, [0.5, 1.5])
        assert_raises(IndexError, a.__getitem__, (['1', '2'],))
        a.__getitem__([])