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
@pytest.mark.parametrize('table_type, dtype', [(ht.Float64HashTable, np.float64), (ht.Float32HashTable, np.float32), (ht.Complex128HashTable, np.complex128), (ht.Complex64HashTable, np.complex64)])
class TestHashTableWithNans:

    def test_get_set_contains_len(self, table_type, dtype):
        index = float('nan')
        table = table_type()
        assert index not in table
        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42
        table.set_item(index, 41)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 41

    def test_map_locations(self, table_type, dtype):
        N = 10
        table = table_type()
        keys = np.full(N, np.nan, dtype=dtype)
        table.map_locations(keys)
        assert len(table) == 1
        assert table.get_item(np.nan) == N - 1

    def test_unique(self, table_type, dtype):
        N = 1020
        table = table_type()
        keys = np.full(N, np.nan, dtype=dtype)
        unique = table.unique(keys)
        assert np.all(np.isnan(unique)) and len(unique) == 1