import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
class TestMultiply:

    def test_seq_repeat(self):
        accepted_types = set(np.typecodes['AllInteger'])
        deprecated_types = {'?'}
        forbidden_types = set(np.typecodes['All']) - accepted_types - deprecated_types
        forbidden_types -= {'V'}
        for seq_type in (list, tuple):
            seq = seq_type([1, 2, 3])
            for numpy_type in accepted_types:
                i = np.dtype(numpy_type).type(2)
                assert_equal(seq * i, seq * int(i))
                assert_equal(i * seq, int(i) * seq)
            for numpy_type in deprecated_types:
                i = np.dtype(numpy_type).type()
                assert_equal(assert_warns(DeprecationWarning, operator.mul, seq, i), seq * int(i))
                assert_equal(assert_warns(DeprecationWarning, operator.mul, i, seq), int(i) * seq)
            for numpy_type in forbidden_types:
                i = np.dtype(numpy_type).type()
                assert_raises(TypeError, operator.mul, seq, i)
                assert_raises(TypeError, operator.mul, i, seq)

    def test_no_seq_repeat_basic_array_like(self):

        class ArrayLike:

            def __init__(self, arr):
                self.arr = arr

            def __array__(self):
                return self.arr
        for arr_like in (ArrayLike(np.ones(3)), memoryview(np.ones(3))):
            assert_array_equal(arr_like * np.float32(3.0), np.full(3, 3.0))
            assert_array_equal(np.float32(3.0) * arr_like, np.full(3, 3.0))
            assert_array_equal(arr_like * np.int_(3), np.full(3, 3))
            assert_array_equal(np.int_(3) * arr_like, np.full(3, 3))