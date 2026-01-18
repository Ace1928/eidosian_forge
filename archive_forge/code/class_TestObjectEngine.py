import re
import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
class TestObjectEngine:
    engine_type = libindex.ObjectEngine
    dtype = np.object_
    values = list('abc')

    def test_is_monotonic(self):
        num = 1000
        arr = np.array(['a'] * num + ['a'] * num + ['c'] * num, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.is_monotonic_increasing is True
        assert engine.is_monotonic_decreasing is False
        engine = self.engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is True
        arr = np.array(['a'] * num + ['b'] * num + ['a'] * num, dtype=self.dtype)
        engine = self.engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is False

    def test_is_unique(self):
        arr = np.array(self.values, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.is_unique is True
        arr = np.array(['a', 'b', 'a'], dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.is_unique is False

    def test_get_loc(self):
        arr = np.array(self.values, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.get_loc('b') == 1
        num = 1000
        arr = np.array(['a'] * num + ['b'] * num + ['c'] * num, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.get_loc('b') == slice(1000, 2000)
        arr = np.array(self.values * num, dtype=self.dtype)
        engine = self.engine_type(arr)
        expected = np.array([False, True, False] * num, dtype=bool)
        result = engine.get_loc('b')
        assert (result == expected).all()