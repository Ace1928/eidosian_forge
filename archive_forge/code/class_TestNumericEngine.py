import re
import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
class TestNumericEngine:

    def test_is_monotonic(self, numeric_indexing_engine_type_and_dtype):
        engine_type, dtype = numeric_indexing_engine_type_and_dtype
        num = 1000
        arr = np.array([1] * num + [2] * num + [3] * num, dtype=dtype)
        engine = engine_type(arr)
        assert engine.is_monotonic_increasing is True
        assert engine.is_monotonic_decreasing is False
        engine = engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is True
        arr = np.array([1] * num + [2] * num + [1] * num, dtype=dtype)
        engine = engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is False

    def test_is_unique(self, numeric_indexing_engine_type_and_dtype):
        engine_type, dtype = numeric_indexing_engine_type_and_dtype
        arr = np.array([1, 3, 2], dtype=dtype)
        engine = engine_type(arr)
        assert engine.is_unique is True
        arr = np.array([1, 2, 1], dtype=dtype)
        engine = engine_type(arr)
        assert engine.is_unique is False

    def test_get_loc(self, numeric_indexing_engine_type_and_dtype):
        engine_type, dtype = numeric_indexing_engine_type_and_dtype
        arr = np.array([1, 2, 3], dtype=dtype)
        engine = engine_type(arr)
        assert engine.get_loc(2) == 1
        num = 1000
        arr = np.array([1] * num + [2] * num + [3] * num, dtype=dtype)
        engine = engine_type(arr)
        assert engine.get_loc(2) == slice(1000, 2000)
        arr = np.array([1, 2, 3] * num, dtype=dtype)
        engine = engine_type(arr)
        expected = np.array([False, True, False] * num, dtype=bool)
        result = engine.get_loc(2)
        assert (result == expected).all()