import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
class TestPrivateCategoricalAPI:

    def test_codes_immutable(self):
        c = Categorical(['a', 'b', 'c', 'a', np.nan])
        exp = np.array([0, 1, 2, 0, -1], dtype='int8')
        tm.assert_numpy_array_equal(c.codes, exp)
        msg = "property 'codes' of 'Categorical' object has no setter" if PY311 else "can't set attribute"
        with pytest.raises(AttributeError, match=msg):
            c.codes = np.array([0, 1, 2, 0, 1], dtype='int8')
        codes = c.codes
        with pytest.raises(ValueError, match='assignment destination is read-only'):
            codes[4] = 1
        c[4] = 'a'
        exp = np.array([0, 1, 2, 0, 0], dtype='int8')
        tm.assert_numpy_array_equal(c.codes, exp)
        c._codes[4] = 2
        exp = np.array([0, 1, 2, 0, 2], dtype='int8')
        tm.assert_numpy_array_equal(c.codes, exp)

    @pytest.mark.parametrize('codes, old, new, expected', [([0, 1], ['a', 'b'], ['a', 'b'], [0, 1]), ([0, 1], ['b', 'a'], ['b', 'a'], [0, 1]), ([0, 1], ['a', 'b'], ['b', 'a'], [1, 0]), ([0, 1], ['b', 'a'], ['a', 'b'], [1, 0]), ([0, 1, 0, 1], ['a', 'b'], ['a', 'b', 'c'], [0, 1, 0, 1]), ([0, 1, 2, 2], ['a', 'b', 'c'], ['a', 'b'], [0, 1, -1, -1]), ([0, 1, -1], ['a', 'b', 'c'], ['a', 'b', 'c'], [0, 1, -1]), ([0, 1, -1], ['a', 'b', 'c'], ['b'], [-1, 0, -1]), ([0, 1, -1], ['a', 'b', 'c'], ['d'], [-1, -1, -1]), ([0, 1, -1], ['a', 'b', 'c'], [], [-1, -1, -1]), ([-1, -1], [], ['a', 'b'], [-1, -1]), ([1, 0], ['b', 'a'], ['a', 'b'], [0, 1])])
    def test_recode_to_categories(self, codes, old, new, expected):
        codes = np.asanyarray(codes, dtype=np.int8)
        expected = np.asanyarray(expected, dtype=np.int8)
        old = Index(old)
        new = Index(new)
        result = recode_for_categories(codes, old, new)
        tm.assert_numpy_array_equal(result, expected)

    def test_recode_to_categories_large(self):
        N = 1000
        codes = np.arange(N)
        old = Index(codes)
        expected = np.arange(N - 1, -1, -1, dtype=np.int16)
        new = Index(expected)
        result = recode_for_categories(codes, old, new)
        tm.assert_numpy_array_equal(result, expected)