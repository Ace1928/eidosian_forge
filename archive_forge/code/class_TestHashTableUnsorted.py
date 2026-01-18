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
class TestHashTableUnsorted:

    def test_string_hashtable_set_item_signature(self):
        tbl = ht.StringHashTable()
        tbl.set_item('key', 1)
        assert tbl.get_item('key') == 1
        with pytest.raises(TypeError, match="'key' has incorrect type"):
            tbl.set_item(4, 6)
        with pytest.raises(TypeError, match="'val' has incorrect type"):
            tbl.get_item(4)

    def test_lookup_nan(self, writable):
        xs = np.array([2.718, 3.14, np.nan, -7, 5, 2, 3])
        xs.setflags(write=writable)
        m = ht.Float64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    def test_add_signed_zeros(self):
        N = 4
        m = ht.Float64HashTable(N)
        m.set_item(0.0, 0)
        m.set_item(-0.0, 0)
        assert len(m) == 1

    def test_add_different_nans(self):
        NAN1 = struct.unpack('d', struct.pack('=Q', 9221120237041090560))[0]
        NAN2 = struct.unpack('d', struct.pack('=Q', 9221120237041090561))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        m = ht.Float64HashTable()
        m.set_item(NAN1, 0)
        m.set_item(NAN2, 0)
        assert len(m) == 1

    def test_lookup_overflow(self, writable):
        xs = np.array([1, 2, 2 ** 63], dtype=np.uint64)
        xs.setflags(write=writable)
        m = ht.UInt64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    @pytest.mark.parametrize('nvals', [0, 10])
    @pytest.mark.parametrize('htable, uniques, dtype, safely_resizes', [(ht.PyObjectHashTable, ht.ObjectVector, 'object', False), (ht.StringHashTable, ht.ObjectVector, 'object', True), (ht.Float64HashTable, ht.Float64Vector, 'float64', False), (ht.Int64HashTable, ht.Int64Vector, 'int64', False), (ht.Int32HashTable, ht.Int32Vector, 'int32', False), (ht.UInt64HashTable, ht.UInt64Vector, 'uint64', False)])
    def test_vector_resize(self, writable, htable, uniques, dtype, safely_resizes, nvals):
        vals = np.array(range(1000), dtype=dtype)
        vals.setflags(write=writable)
        htable = htable()
        uniques = uniques()
        htable.get_labels(vals[:nvals], uniques, 0, -1)
        tmp = uniques.to_array()
        oldshape = tmp.shape
        if safely_resizes:
            htable.get_labels(vals, uniques, 0, -1)
        else:
            with pytest.raises(ValueError, match='external reference.*'):
                htable.get_labels(vals, uniques, 0, -1)
        uniques.to_array()
        assert tmp.shape == oldshape

    @pytest.mark.parametrize('hashtable', [ht.PyObjectHashTable, ht.StringHashTable, ht.Float64HashTable, ht.Int64HashTable, ht.Int32HashTable, ht.UInt64HashTable])
    def test_hashtable_large_sizehint(self, hashtable):
        size_hint = np.iinfo(np.uint32).max + 1
        hashtable(size_hint=size_hint)