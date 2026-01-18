from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.complex128, np.complex64])
class TestHelpFunctionsWithNans:

    def test_value_count(self, dtype):
        values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        keys, counts, _ = ht.value_count(values, True)
        assert len(keys) == 0
        keys, counts, _ = ht.value_count(values, False)
        assert len(keys) == 1 and np.all(np.isnan(keys))
        assert counts[0] == 3

    def test_duplicated_first(self, dtype):
        values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        result = ht.duplicated(values)
        expected = np.array([False, True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_yes(self, dtype):
        arr = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        values = np.array([np.nan, np.nan], dtype=dtype)
        result = ht.ismember(arr, values)
        expected = np.array([True, True, True], dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_no(self, dtype):
        arr = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        values = np.array([1], dtype=dtype)
        result = ht.ismember(arr, values)
        expected = np.array([False, False, False], dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_mode(self, dtype):
        values = np.array([42, np.nan, np.nan, np.nan], dtype=dtype)
        assert ht.mode(values, True)[0] == 42
        assert np.isnan(ht.mode(values, False)[0])