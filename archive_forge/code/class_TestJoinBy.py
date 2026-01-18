import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
class TestJoinBy:

    def setup_method(self):
        self.a = np.array(list(zip(np.arange(10), np.arange(50, 60), np.arange(100, 110))), dtype=[('a', int), ('b', int), ('c', int)])
        self.b = np.array(list(zip(np.arange(5, 15), np.arange(65, 75), np.arange(100, 110))), dtype=[('a', int), ('b', int), ('d', int)])

    def test_inner_join(self):
        a, b = (self.a, self.b)
        test = join_by('a', a, b, jointype='inner')
        control = np.array([(5, 55, 65, 105, 100), (6, 56, 66, 106, 101), (7, 57, 67, 107, 102), (8, 58, 68, 108, 103), (9, 59, 69, 109, 104)], dtype=[('a', int), ('b1', int), ('b2', int), ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_join(self):
        a, b = (self.a, self.b)
        join_by(('a', 'b'), a, b)
        np.array([(5, 55, 105, 100), (6, 56, 106, 101), (7, 57, 107, 102), (8, 58, 108, 103), (9, 59, 109, 104)], dtype=[('a', int), ('b', int), ('c', int), ('d', int)])

    def test_join_subdtype(self):
        foo = np.array([(1,)], dtype=[('key', int)])
        bar = np.array([(1, np.array([1, 2, 3]))], dtype=[('key', int), ('value', 'uint16', 3)])
        res = join_by('key', foo, bar)
        assert_equal(res, bar.view(ma.MaskedArray))

    def test_outer_join(self):
        a, b = (self.a, self.b)
        test = join_by(('a', 'b'), a, b, 'outer')
        control = ma.array([(0, 50, 100, -1), (1, 51, 101, -1), (2, 52, 102, -1), (3, 53, 103, -1), (4, 54, 104, -1), (5, 55, 105, -1), (5, 65, -1, 100), (6, 56, 106, -1), (6, 66, -1, 101), (7, 57, 107, -1), (7, 67, -1, 102), (8, 58, 108, -1), (8, 68, -1, 103), (9, 59, 109, -1), (9, 69, -1, 104), (10, 70, -1, 105), (11, 71, -1, 106), (12, 72, -1, 107), (13, 73, -1, 108), (14, 74, -1, 109)], mask=[(0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)], dtype=[('a', int), ('b', int), ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_leftouter_join(self):
        a, b = (self.a, self.b)
        test = join_by(('a', 'b'), a, b, 'leftouter')
        control = ma.array([(0, 50, 100, -1), (1, 51, 101, -1), (2, 52, 102, -1), (3, 53, 103, -1), (4, 54, 104, -1), (5, 55, 105, -1), (6, 56, 106, -1), (7, 57, 107, -1), (8, 58, 108, -1), (9, 59, 109, -1)], mask=[(0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1)], dtype=[('a', int), ('b', int), ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_different_field_order(self):
        a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
        b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
        j = join_by(['c', 'b'], a, b, jointype='inner', usemask=False)
        assert_equal(j.dtype.names, ['b', 'c', 'a1', 'a2'])

    def test_duplicate_keys(self):
        a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
        b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
        assert_raises(ValueError, join_by, ['a', 'b', 'b'], a, b)

    def test_same_name_different_dtypes_key(self):
        a_dtype = np.dtype([('key', 'S5'), ('value', '<f4')])
        b_dtype = np.dtype([('key', 'S10'), ('value', '<f4')])
        expected_dtype = np.dtype([('key', 'S10'), ('value1', '<f4'), ('value2', '<f4')])
        a = np.array([('Sarah', 8.0), ('John', 6.0)], dtype=a_dtype)
        b = np.array([('Sarah', 10.0), ('John', 7.0)], dtype=b_dtype)
        res = join_by('key', a, b)
        assert_equal(res.dtype, expected_dtype)

    def test_same_name_different_dtypes(self):
        a_dtype = np.dtype([('key', 'S10'), ('value', '<f4')])
        b_dtype = np.dtype([('key', 'S10'), ('value', '<f8')])
        expected_dtype = np.dtype([('key', '|S10'), ('value1', '<f4'), ('value2', '<f8')])
        a = np.array([('Sarah', 8.0), ('John', 6.0)], dtype=a_dtype)
        b = np.array([('Sarah', 10.0), ('John', 7.0)], dtype=b_dtype)
        res = join_by('key', a, b)
        assert_equal(res.dtype, expected_dtype)

    def test_subarray_key(self):
        a_dtype = np.dtype([('pos', int, 3), ('f', '<f4')])
        a = np.array([([1, 1, 1], np.pi), ([1, 2, 3], 0.0)], dtype=a_dtype)
        b_dtype = np.dtype([('pos', int, 3), ('g', '<f4')])
        b = np.array([([1, 1, 1], 3), ([3, 2, 1], 0.0)], dtype=b_dtype)
        expected_dtype = np.dtype([('pos', int, 3), ('f', '<f4'), ('g', '<f4')])
        expected = np.array([([1, 1, 1], np.pi, 3)], dtype=expected_dtype)
        res = join_by('pos', a, b)
        assert_equal(res.dtype, expected_dtype)
        assert_equal(res, expected)

    def test_padded_dtype(self):
        dt = np.dtype('i1,f4', align=True)
        dt.names = ('k', 'v')
        assert_(len(dt.descr), 3)
        a = np.array([(1, 3), (3, 2)], dt)
        b = np.array([(1, 1), (2, 2)], dt)
        res = join_by('k', a, b)
        expected_dtype = np.dtype([('k', 'i1'), ('v1', 'f4'), ('v2', 'f4')])
        assert_equal(res.dtype, expected_dtype)